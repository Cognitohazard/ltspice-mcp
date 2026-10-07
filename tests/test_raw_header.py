"""Complete RAW layout checks; builders are synthetic, recordings are labelled."""

from __future__ import annotations

import os
import struct
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from ltspice_mcp.lib.raw_header import (
    RawDialectError,
    RawHeaderError,
    RawLayoutError,
    RawLimitError,
    RawLimits,
    preflight_raw,
)

FIXTURES = Path(__file__).parent / "fixtures"
# Test admission budgets, not measured production defaults.
LIMITS = RawLimits(
    input_bytes=2_000_000,
    header_bytes=64_000,
    line_bytes=16_000,
    plots=16,
    variables=128,
    points=10_000,
    numeric_bytes=2_000_000,
)


def synthetic_header(
    *,
    variables: tuple[tuple[str, str], ...] = (("time", "time"), ("V(out)", "voltage")),
    points: int = 2,
    flags: str = "real",
    command: str | None = "ngspice-42",
    plot: str = "Transient Analysis",
    kind: str = "Binary",
    extra: str = "",
) -> str:
    return (
        "Title: synthetic\nDate: synthetic\n"
        f"Plotname: {plot}\nFlags: {flags}\n"
        f"No. Variables: {len(variables)}\nNo. Points: {points}\n"
        + (f"Command: {command}\n" if command is not None else "")
        + extra
        + "Variables:\n"
        + "".join(f"\t{i}\t{name}\t{declared}\n" for i, (name, declared) in enumerate(variables))
        + f"{kind}:\n"
    )


def write_raw(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "synthetic.raw"
    path.write_bytes(data)
    return path


def synthetic_binary(**kwargs) -> bytes:
    return synthetic_header(**kwargs).encode("ascii") + struct.pack("<4d", 0, 1, 1, 2)


@pytest.mark.parametrize(
    ("name", "points", "widths"),
    [
        ("ltspice_tran_rc.raw", 221, (8, 4, 4, 4, 4, 4)),
        ("ltspice_ac_rc.raw", 81, (16,) * 6),
        ("ltspice_dc_div.raw", 11, (8, 4, 4, 4, 4, 4)),
        ("ltspice_noise_rc.raw", 121, (8, 4, 4, 4, 4)),
        ("op_extreme_node.raw", 1, (8, 4, 4)),
        ("ltspice_step_tran.raw", 3154, (8,) + (4,) * 8),
        ("ltspice_step_ac.raw", 243, (16,) * 6),
    ],
)
def test_recorded_ltspice_binary(name: str, points: int, widths: tuple[int, ...]):
    path = FIXTURES / name
    result = preflight_raw(path, limits=LIMITS)
    (plot,) = result.plots
    assert result.size_bytes == path.stat().st_size
    assert (plot.dialect, plot.dialect_evidence) == ("ltspice", ("header:Command",))
    assert plot.encoding == "utf-16-le"
    assert plot.point_count == points
    assert plot.variable_count == len(widths)
    assert plot.value_bytes == widths
    assert plot.payload_end == result.size_bytes
    assert plot.payload_bytes == plot.numeric_bytes == points * sum(widths)
    assert plot.payload_end - plot.payload_offset == plot.payload_bytes
    assert [v.index for v in plot.variables] == list(range(len(widths)))
    if "step" in name:
        assert "stepped" in plot.flags
    if "noise" in name:
        assert plot.plot_name == "Noise Spectral Density - (V/Hz½ or A/Hz½)"
    if "tran_rc" in name:
        assert (plot.variables[2].name, plot.variables[2].declared_type) == ("V(out)", "voltage")
        assert dict(plot.fields)["Offset"] == "0.0000000000000000e+00"


def test_recorded_ngspice_ascii_two_plots():
    result = preflight_raw(FIXTURES / "ngspice_noise_2plot.raw", limits=LIMITS)
    first, second = result.plots
    assert [p.plot_name for p in result.plots] == [
        "Noise Spectral Density Curves",
        "Integrated Noise",
    ]
    assert (first.point_count, second.point_count) == (301, 1)
    assert first.variables[0].attributes == ("grid=3",)
    assert first.variables[1].declared_type == "voltage-density"
    assert second.variables[0].name == "v(onoise_total)"
    assert first.payload_end == second.header_offset
    assert second.payload_end == result.size_bytes
    assert result.numeric_bytes == 301 * 3 * 8 + 2 * 8


@pytest.mark.parametrize("kind", ["ascii", "binary"])
@pytest.mark.parametrize("case", ["multi", "disto_op", "disto_two"])
def test_preserved_ngspice_multi_outputs(case: str, kind: str):
    """Optional genuine ngspice-42 recordings, supplied outside version control."""
    directory = os.environ.get("LTSPICE_MCP_TEST_ANALYSIS_RAW_DIR")
    if directory is None:
        pytest.skip("preserved ngspice analysis recordings were not supplied")
    path = Path(directory) / case / kind / "result.raw"
    result = preflight_raw(path, limits=LIMITS, producing_dialect="ngspice")
    expected = {
        "multi": [
            "AC Analysis",
            "Operating Point",
            "Pole-Zero Analysis",
            "Transfer Function",
            "Sensitivity Analysis",
        ],
        "disto_op": [
            "Operating Point",
            "DISTORTION - IM: f1+f2",
            "DISTORTION - IM: f1-f2",
            "DISTORTION - IM: 2f1-f2",
        ],
        "disto_two": [
            "DISTORTION - IM: f1+f2",
            "DISTORTION - IM: f1-f2",
            "DISTORTION - IM: 2f1-f2",
        ],
    }
    assert [p.plot_name for p in result.plots] == expected[case]
    assert all(p.dialect_evidence == ("producing_dialect",) for p in result.plots)
    assert all(
        a.payload_end == b.header_offset
        for a, b in zip(result.plots, result.plots[1:], strict=False)
    )
    assert result.plots[-1].payload_end == result.size_bytes
    if case == "multi":
        assert [p.variable_count for p in result.plots] == [4, 3, 1, 3, 25]
        assert result.plots[2].variables[0].name == "v(pole(1))"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_encoding_and_byte_offsets(tmp_path: Path, encoding: str, bom: bool, newline: str):
    prefix = (
        {"utf-8": b"\xef\xbb\xbf", "utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff"}[encoding]
        if bom
        else b""
    )
    header = synthetic_header().replace("\n", newline)
    data = prefix + header.encode(encoding) + struct.pack("<4d", 0, 1, 1, 2)
    result = preflight_raw(write_raw(tmp_path, data), limits=LIMITS)
    (plot,) = result.plots
    assert plot.encoding == encoding
    assert plot.header_offset == 0
    assert plot.payload_offset == len(prefix) + len(header.encode(encoding))
    assert plot.payload_end == len(data)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_text_values_encoding_and_rows(tmp_path: Path, encoding: str):
    header = synthetic_header(kind="Values")
    body = "0\t\t0.0\r\n\tNaN\r\n\r\n1\t\t1e-3\r\n\t-INF\r\n"
    data = (header + body).encode(encoding)
    (plot,) = preflight_raw(write_raw(tmp_path, data), limits=LIMITS).plots
    assert plot.payload_offset == len(header.encode(encoding))
    assert plot.payload_end == len(data)
    assert plot.storage == "values"
    assert plot.numeric_bytes == 32


@pytest.mark.parametrize("value", ["-1", "0", "1.5", "+2", "NaN", "1_000", "\uff19", "9" * 5000])
@pytest.mark.parametrize("field", ["No. Variables", "No. Points"])
def test_bad_counts(tmp_path: Path, field: str, value: str):
    data = synthetic_binary().replace(f"{field}: 2".encode(), f"{field}: {value}".encode())
    with pytest.raises(RawHeaderError, match=field.replace(".", r"\.")):
        preflight_raw(write_raw(tmp_path, data), limits=LIMITS)


@pytest.mark.parametrize(
    "field", ["No. Variables", "No. Points", "Flags", "Plotname", "Variables"]
)
def test_missing_and_duplicate_fields(tmp_path: Path, field: str):
    data = synthetic_binary()
    line = next(
        line for line in data.splitlines(keepends=True) if line.startswith((field + ":").encode())
    )
    for changed in (data.replace(line, b""), data.replace(line, line * 2)):
        with pytest.raises(RawHeaderError):
            preflight_raw(write_raw(tmp_path, changed), limits=LIMITS)


@pytest.mark.parametrize(
    "row",
    [
        b"\t0\tV(out)\tvoltage",
        b"\t2\tV(out)\tvoltage",
        b"\t-1\tV(out)\tvoltage",
        b"\t1\tV(out)",
        b"\t1\t\tvoltage",
        b"\t1\ttime\tvoltage",
    ],
)
def test_invalid_variable_rows(tmp_path: Path, row: bytes):
    data = synthetic_binary().replace(b"\t1\tV(out)\tvoltage", row)
    with pytest.raises(RawHeaderError, match="variable"):
        preflight_raw(write_raw(tmp_path, data), limits=LIMITS)


@pytest.mark.parametrize(
    "later",
    [
        b"Title: torn",
        synthetic_binary(points=999),
        synthetic_binary().replace(b"No. Points: 2", b"No. Points: -1"),
        synthetic_binary()[:-1],
    ],
)
def test_malformed_later_plot_refuses_entire_input(tmp_path: Path, later: bytes):
    with pytest.raises(RawHeaderError, match="plot 1"):
        preflight_raw(write_raw(tmp_path, synthetic_binary() + later), limits=LIMITS)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
def test_long_lines_are_bounded_and_long_permitted_header_works(tmp_path: Path, encoding: str):
    header = synthetic_header(extra="Note: " + "x" * 2000 + "\n")
    data = header.encode(encoding) + bytes(32)
    path = write_raw(tmp_path, data)
    with pytest.raises(RawLimitError, match="line_bytes"):
        preflight_raw(path, limits=replace(LIMITS, line_bytes=256))
    (plot,) = preflight_raw(path, limits=LIMITS).plots
    assert dict(plot.fields)["Note"] == "x" * 2000


def test_header_limit_is_cumulative_across_plots(tmp_path: Path):
    header_size = len(synthetic_header().encode())
    path = write_raw(tmp_path, synthetic_binary() * 2)
    with pytest.raises(RawLimitError, match="header_bytes"):
        preflight_raw(path, limits=replace(LIMITS, header_bytes=header_size * 2 - 1))
    assert (
        len(preflight_raw(path, limits=replace(LIMITS, header_bytes=header_size * 2)).plots) == 2
    )


@pytest.mark.parametrize("name", [field.name for field in fields(RawLimits)])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("inf"), None])
def test_limits_require_finite_positive_integers(name: str, value):
    with pytest.raises(ValueError, match=name):
        replace(LIMITS, **{name: value})


@pytest.mark.parametrize(
    ("field", "limit"),
    [("input_bytes", 1), ("plots", 1), ("variables", 1), ("points", 1), ("numeric_bytes", 63)],
)
def test_each_admission_limit(tmp_path: Path, field: str, limit: int):
    path = write_raw(tmp_path, synthetic_binary() * 2)
    with pytest.raises(RawLimitError, match=field):
        preflight_raw(path, limits=replace(LIMITS, **{field: limit}))


def test_payload_cannot_supply_dialect(tmp_path: Path):
    # Synthetic binary bytes that look like header lines are not header evidence.
    spoof = b"Command: LTspice\nTitle: ngspice\n"
    header = synthetic_header(points=4)
    body = spoof.ljust(64, b"\0")
    result = preflight_raw(write_raw(tmp_path, header.encode() + body), limits=LIMITS)
    assert result.plots[0].dialect == "ngspice"
    assert result.plots[0].dialect_evidence == ("header:Command",)
    no_writer = header.replace("Command: ngspice-42\n", "").encode() + body
    with pytest.raises(RawDialectError, match="ambiguous"):
        preflight_raw(write_raw(tmp_path, no_writer), limits=LIMITS)
    assert (
        preflight_raw(write_raw(tmp_path, no_writer), limits=LIMITS, dialect="ngspice")
        .plots[0]
        .dialect
        == "ngspice"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dialect": "ltspice"},
        {"producing_dialect": "ltspice"},
        {"dialect": "ngspice", "producing_dialect": "ltspice"},
    ],
)
def test_contradictory_dialect(tmp_path: Path, kwargs):
    with pytest.raises(RawDialectError, match="conflict"):
        preflight_raw(write_raw(tmp_path, synthetic_binary()), limits=LIMITS, **kwargs)


def test_later_writer_cannot_conflict(tmp_path: Path):
    later = synthetic_binary(command="LTspice", flags="real double")
    with pytest.raises(RawDialectError, match=r"plot 1.*conflict"):
        preflight_raw(write_raw(tmp_path, synthetic_binary() + later), limits=LIMITS)


@pytest.mark.parametrize("command", ["ngspice with LTspice", "notngspice", "unknown writer"])
def test_ambiguous_command(tmp_path: Path, command: str):
    with pytest.raises(RawDialectError, match="ambiguous"):
        preflight_raw(write_raw(tmp_path, synthetic_binary(command=command)), limits=LIMITS)


def test_explicit_and_producing_evidence(tmp_path: Path):
    result = preflight_raw(
        write_raw(tmp_path, synthetic_binary()),
        limits=LIMITS,
        dialect="NGSPICE",
        producing_dialect="ngspice",
    )
    assert result.plots[0].dialect_evidence == (
        "explicit_dialect",
        "producing_dialect",
        "header:Command",
    )


def test_legacy_inference_has_specific_evidence(tmp_path: Path):
    lt = (FIXTURES / "ltspice_tran_rc.raw").read_bytes()
    lt = lt.replace("Command: Linear Technology Corporation LTspice\n".encode("utf-16-le"), b"")
    (plot,) = preflight_raw(write_raw(tmp_path, lt), limits=LIMITS).plots
    assert (plot.dialect, plot.dialect_evidence) == ("ltspice", ("legacy:utf16-offset-forward",))
    ng = (FIXTURES / "ngspice_noise_2plot.raw").read_bytes()
    ng = b"".join(
        line for line in ng.splitlines(keepends=True) if not line.startswith(b"Command:")
    )
    plots = preflight_raw(write_raw(tmp_path, ng), limits=LIMITS).plots
    assert plots[0].dialect_evidence == ("legacy:ngspice-noise-grid",)
    assert plots[1].dialect_evidence == ("previous_plot",)


@pytest.mark.parametrize(
    ("command", "flags", "widths"),
    [
        ("LTspice", "real", (8, 4)),
        ("LTspice", "real double", (8, 8)),
        ("LTspice", "real fastaccess", (8, 4)),
        ("LTspice", "complex fastaccess", (16, 16)),
        ("ngspice", "complex", (16, 16)),
        ("QSPICE", "complex", (8, 16)),
        ("QSPICE", "real", (8, 8)),
        ("Xyce", "real", (8, 8)),
        ("Xyce", "complex", (16, 16)),
    ],
)
def test_source_defined_allowlisted_layouts(
    tmp_path: Path, command: str, flags: str, widths: tuple[int, ...]
):
    header = synthetic_header(
        command=command,
        flags=flags,
        plot="AC Analysis" if "complex" in flags else "Transient Analysis",
    )
    body = bytes(2 * sum(widths))
    (plot,) = preflight_raw(write_raw(tmp_path, header.encode() + body), limits=LIMITS).plots
    assert plot.value_bytes == widths
    assert plot.storage_order == ("trace" if "fastaccess" in flags else "point")
    assert plot.numeric_bytes == len(body)


@pytest.mark.parametrize(
    ("command", "flags", "plot"),
    [
        ("LTspice", "real compressed", "Transient Analysis"),
        ("ngspice", "real unpadded", "Transient Analysis"),
        ("ngspice", "real fastaccess", "Transient Analysis"),
        ("LTspice", "complex double", "AC Analysis"),
        ("LTspice", "real", "AC Analysis"),
        ("LTspice", "real", "Unknown analysis"),
    ],
)
def test_unsupported_layouts_have_precise_refusals(
    tmp_path: Path, command: str, flags: str, plot: str
):
    with pytest.raises(RawLayoutError, match=r"layout|flag|AC Analysis"):
        preflight_raw(
            write_raw(tmp_path, synthetic_binary(command=command, flags=flags, plot=plot)),
            limits=LIMITS,
        )


@pytest.mark.parametrize("flags", ["", "real complex", "real real", "forward"])
def test_invalid_flags(tmp_path: Path, flags: str):
    with pytest.raises(RawHeaderError, match=r"[Ff]lags"):
        preflight_raw(write_raw(tmp_path, synthetic_binary(flags=flags)), limits=LIMITS)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"Title:",
        b"Title: invalid\xff\n",
        b"\xff\xfeT\0i",
        synthetic_binary()[:-1],
        synthetic_binary() + b"garbage",
        synthetic_binary() + b"\n",
        synthetic_header().replace("Binary:\n", "Binary:").encode(),
    ],
)
def test_encoding_truncation_and_trailing_garbage(tmp_path: Path, data: bytes):
    with pytest.raises(RawHeaderError):
        preflight_raw(write_raw(tmp_path, data), limits=LIMITS)


@pytest.mark.parametrize(
    "body",
    [
        "0\t\t0\n\t1\n1\t\t1\n",
        "0\t\t0\n\t1\n2\t\t1\n\t2\n",
        "0\t\t0\n\tCommand: LTspice\n1\t\t1\n\t2\n",
        "0\t\t0\n\t1\n1\t\t1\n\t2\n\t3\n",
        "0\t\t0\n\t1\n1\t\t1\n\t2",
        "0\t\t0\n\t1\n1\t\t1\n\t2,3\n",
    ],
)
def test_malformed_text_payload(tmp_path: Path, body: str):
    path = write_raw(tmp_path, (synthetic_header(kind="Values") + body).encode())
    with pytest.raises(RawHeaderError):
        preflight_raw(path, limits=LIMITS)


def test_text_complex_and_line_limit(tmp_path: Path):
    header = synthetic_header(kind="Values", flags="complex")
    body = "0\t\t1,-2\n\t3e-4,5e-6\n1\t\t2,0\n\tNaN,inf\n"
    (plot,) = preflight_raw(write_raw(tmp_path, (header + body).encode()), limits=LIMITS).plots
    assert plot.value_bytes == (16, 16)
    assert plot.numeric_bytes == 64
    long_body = body.replace("3e-4", "3" * 300)
    with pytest.raises(RawLimitError, match="line_bytes"):
        preflight_raw(
            write_raw(tmp_path, (header + long_body).encode()),
            limits=replace(LIMITS, line_bytes=256),
        )


def test_metadata_is_frozen(tmp_path: Path):
    result = preflight_raw(write_raw(tmp_path, synthetic_binary()), limits=LIMITS)
    with pytest.raises(FrozenInstanceError):
        result.plots[0].point_count = 3  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_later_text_plot_is_fully_checked(tmp_path: Path, encoding: str):
    header = synthetic_header(kind="Values")
    good = header + "0\t\t0\n\t1\n1\t\t1\n\t2\n"
    bad = header + "0\t\t0\n\t1\n1\t\t1\n\tCommand: LTspice\n"
    with pytest.raises(RawHeaderError, match=r"plot 1.*numeric text payload"):
        preflight_raw(write_raw(tmp_path, (good + bad).encode(encoding)), limits=LIMITS)


def test_numeric_budget_refuses_before_payload_walk(tmp_path: Path):
    # Deliberately missing data: the declared allocation must be refused first.
    header = synthetic_header(kind="Values", points=5000)
    with pytest.raises(RawLimitError, match="numeric_bytes"):
        preflight_raw(
            write_raw(tmp_path, header.encode()),
            limits=replace(LIMITS, numeric_bytes=100),
        )


@pytest.mark.parametrize("location", ["Title: synthetic", "Note: neutral"])
def test_only_command_field_supplies_writer_evidence(tmp_path: Path, location: str):
    header = synthetic_header(extra="Note: neutral\n")
    data = header.replace(location, location + " LTspice QSPICE Xyce").encode() + bytes(32)
    (plot,) = preflight_raw(write_raw(tmp_path, data), limits=LIMITS).plots
    assert (plot.dialect, plot.dialect_evidence) == ("ngspice", ("header:Command",))


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_surrogates_are_decoded_strictly(tmp_path: Path, encoding: str):
    header = synthetic_header().encode(encoding)
    invalid = header.replace(
        "synthetic".encode(encoding), "\ud800".encode(encoding, errors="surrogatepass"), 1
    )
    with pytest.raises(RawHeaderError, match="encoding"):
        preflight_raw(write_raw(tmp_path, invalid + bytes(32)), limits=LIMITS)


@pytest.mark.parametrize("value", ["", "unknown", "ngspice ", 42])
@pytest.mark.parametrize("name", ["dialect", "producing_dialect"])
def test_invalid_dialect_argument(tmp_path: Path, name: str, value):
    with pytest.raises(RawDialectError, match=name):
        preflight_raw(write_raw(tmp_path, synthetic_binary()), limits=LIMITS, **{name: value})


def test_missing_and_nonregular_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        preflight_raw(tmp_path / "absent.raw", limits=LIMITS)
    with pytest.raises(RawHeaderError, match="regular file"):
        preflight_raw(tmp_path, limits=LIMITS)


def test_unterminated_long_header_reports_line_limit(tmp_path: Path):
    data = b"Title: " + b"x" * 100_000
    with pytest.raises(RawLimitError, match="line_bytes"):
        preflight_raw(write_raw(tmp_path, data), limits=replace(LIMITS, line_bytes=256))


def test_text_separators_are_excluded_from_payload_offsets(tmp_path: Path):
    first = synthetic_header(kind="Values") + "0\t\t0\n\t1\n1\t\t1\n\t2\n"
    separator = "\n\t\n"
    data = (first + separator).encode() + synthetic_binary()
    result = preflight_raw(write_raw(tmp_path, data), limits=LIMITS)
    assert result.plots[0].payload_end == len(first.encode())
    assert result.plots[1].header_offset == len((first + separator).encode())


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_line_reader_handles_lf_byte_in_other_code_unit(tmp_path: Path, encoding: str):
    # Neither character is a newline; each embeds an LF byte in one byte order.
    header = synthetic_header(extra="Note: \u0a41\u410a\n")
    (plot,) = preflight_raw(
        write_raw(tmp_path, header.encode(encoding) + bytes(32)), limits=LIMITS
    ).plots
    assert dict(plot.fields)["Note"] == "\u0a41\u410a"
