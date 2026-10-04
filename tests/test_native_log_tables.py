"""Recorded native print blocks, explicit ambiguity and bounded corruption."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.log_decode import LogLimitError, LogLimits, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs

FIXTURES = Path(__file__).parent / "fixtures" / "native_log_tables"
LIMITS = LogLimits(1_000_000, 16_000, 20_000, 50_000, 2_000_000)


def decode(tmp_path, *, case=None, text=None, limits=LIMITS, console=None):
    if case:
        text = (FIXTURES / f"{case}.log").read_text(encoding="utf-8")
    source = tmp_path / "source.log"
    source.write_text(text or "", encoding="utf-8", newline="")
    console_source = None
    if console:
        console_source = tmp_path / "source.exe.log"
        console_source.write_text(console, encoding="utf-8")
    directory = tmp_path / "worker"
    directory.mkdir()
    captured = capture_inputs(
        SourceFiles(log=source, console=console_source),
        directory,
        input_bytes=LIMITS.log_bytes,
        log_bytes=LIMITS.log_bytes,
        require_raw=False,
    )
    return decode_logs(captured, directory, limits=limits)


@pytest.mark.parametrize(
    ("case", "blocks", "rows"),
    [
        ("tf_table", 1, 3),
        ("pz_table", 1, 1),
        ("sens_dc_table", 1, 21),
        ("sens_ac_table", 24, 72),
        ("disto_harm_table", 4, 20),
        ("disto_two_table", 6, 30),
    ],
)
def test_six_genuine_recordings_preserve_physical_print_blocks(tmp_path, case, blocks, rows):
    metadata = decode(tmp_path, case=case)
    section = metadata["native_tables"]
    assert section["status"] == "parsed" and section["error"] is None
    assert section["nonfinite_count"] == 0
    facts = section["value"]
    assert len(facts) == blocks
    assert sum(len(block.get("entries", block.get("rows", []))) for block in facts) == rows
    previous = 0
    for ordinal, block in enumerate(facts):
        assert block["ordinal"] == ordinal
        assert previous < block["line_start"] <= block["line_end"]
        previous = block["line_end"]
        assert block["analysis_extent"] == "unknown"
        assert "plot_id" not in block and "step_index" not in block
        if block["layout"] == "scalar_print":
            assert block["axis"] is None
            assert all(entry["unit"] is None for entry in block["entries"])
        else:
            assert block["coordinate"] == {
                "label": "frequency",
                "unit": None,
                "convention": "printed",
            }
            assert block["column"]["unit"] is None
    assert DecodedLog(metadata).as_dict() == metadata
    json.dumps(metadata, allow_nan=False)


def test_tf_and_pz_keep_exact_quantity_labels_and_components(tmp_path):
    tf = DecodedLog(decode(tmp_path, case="tf_table")).value("native_tables")[0]
    assert [(e["label"], e["real"], e["imag"]) for e in tf["entries"]] == [
        ("transfer_function", 0.6666666666666666, None),
        ("output_impedance_at_v(out)", 666.6666666666666, None),
        ("v1#input_impedance", 3000.0, None),
    ]
    pz_root = tmp_path / "pz"
    pz_root.mkdir()
    pz = DecodedLog(decode(pz_root, case="pz_table")).value("native_tables")[0]
    assert pz["entries"] == [
        {
            "line": 18,
            "label": "all",
            "representation": "complex",
            "real": -1000.0,
            "imag": 0.0,
            "unit": None,
        }
    ]
    assert pz["printed_analysis_label"] is None
    assert pz["listing_current"]["id"] == "pz1"


def test_sensitivity_keeps_scalar_names_signed_zero_and_emitted_axis(tmp_path):
    scalar = DecodedLog(decode(tmp_path, case="sens_dc_table")).value("native_tables")[0]
    entries = scalar["entries"]
    assert entries[0]["label"] == "r1" and entries[0]["real"] == -6.66666000037388e-4
    assert entries[1]["label"] == "r1:bv_max"
    assert math.copysign(1, entries[1]["real"]) == -1
    ac_root = tmp_path / "ac"
    ac_root.mkdir()
    blocks = DecodedLog(decode(ac_root, case="sens_ac_table")).value("native_tables")
    assert blocks[0]["column"]["label"] == "r1"
    assert blocks[-1]["column"]["label"] == "v1_acmag"
    assert [r["frequency"] for r in blocks[0]["rows"]] == [
        100.0,
        6666.666666666667,
        444444.4444444445,
    ]
    assert blocks[0]["rows"][0]["real"] == -2.22222000012463e-4


@pytest.mark.parametrize(
    ("case", "labels", "last"),
    [
        (
            "disto_harm_table",
            ["DISTORTION - 2nd harmonic"] * 2 + ["DISTORTION - 3rd harmonic"] * 2,
            -1.25315226780583e-8,
        ),
        (
            "disto_two_table",
            ["DISTORTION - IM: f1+f2"] * 2
            + ["DISTORTION - IM: f1-f2"] * 2
            + ["DISTORTION - IM: 2f1-f2"] * 2,
            -1.87972840170875e-8,
        ),
    ],
)
def test_distortion_current_listing_is_not_print_identity(tmp_path, case, labels, last):
    blocks = DecodedLog(decode(tmp_path, case=case)).value("native_tables")
    assert [b["printed_analysis_label"] for b in blocks] == labels
    assert blocks[0]["listing_current"]["analysis_label"] != blocks[0]["printed_analysis_label"]
    assert [b["column"]["label"] for b in blocks] == ["out", "v1#branch"] * (len(blocks) // 2)
    assert [r["index"] for r in blocks[0]["rows"]] == [0, 1, 2, 3, 4]
    assert blocks[-1]["rows"][0]["real"] == last


@pytest.mark.parametrize(
    ("case", "closures"),
    [
        ("sens_ac_table", ["form_feed"] * 23 + ["ngspice_done"]),
        ("disto_harm_table", ["form_feed", "next_print_header", "form_feed", "ngspice_done"]),
        (
            "disto_two_table",
            ["form_feed", "next_print_header"] * 2 + ["form_feed", "ngspice_done"],
        ),
    ],
)
def test_block_closure_preserves_literal_page_and_header_boundaries(tmp_path, case, closures):
    blocks = DecodedLog(decode(tmp_path, case=case)).value("native_tables")
    assert [block["closed_by"] for block in blocks] == closures


def test_eof_after_form_feed_keeps_closed_block_without_analysis_completeness(tmp_path):
    body = (FIXTURES / "disto_harm_table.log").read_text(encoding="utf-8")
    body = body[: body.index("\f") + 1] + "\n"
    blocks = DecodedLog(decode(tmp_path, text=body)).value("native_tables")
    assert len(blocks) == 1
    assert blocks[0]["closed_by"] == "form_feed"
    assert blocks[0]["analysis_extent"] == "unknown"


@pytest.mark.parametrize(
    "change",
    ["footer", "row-tail", "comma", "header", "rule", "width", "index", "token", "title", "mixed"],
)
def test_actual_recording_corruption_is_error_not_successful_prefix(tmp_path, change):
    body = (FIXTURES / "disto_harm_table.log").read_text(encoding="utf-8")
    if change == "footer":
        body = body.rsplit("ngspice-42 done", 1)[0]
    elif change == "row-tail":
        body = body[: body.index("1\t1.500000000000000e+02") + 8]
    elif change == "comma":
        body = body.replace(
            "0.000000000000000e+00,\t0.000000000000000e+00", "0.000000000000000e+00,", 1
        )
    elif change == "header":
        body = body[: body.index("Index") + 8]
    elif change == "rule":
        body = body.replace(
            "out                             \n" + "-" * 80,
            "out                             \n",
            1,
        )
    elif change == "width":
        body = body.replace("Index   frequency       out", "Index frequency out extra", 1)
    elif change == "index":
        body = body.replace("1\t1.500000000000000e+02", "4\t1.500000000000000e+02", 1)
    elif change == "token":
        body = body.replace("0.000000000000000e+00,", "1e999junk,", 1)
    elif change == "title":
        body = body.replace("DISTORTION - 2nd harmonic  Thu", "DISTORTION - unrecognized  Thu", 1)
    else:
        body = body.replace("0.000000000000000e+00,\t0.000000000000000e+00", "0.0", 1)
    metadata = decode(tmp_path, text=body)
    section = metadata["native_tables"]
    assert section["status"] == "error" and section["value"] is None
    assert "line" in section["error"]["message"]
    with pytest.raises(ResultError, match="native_tables"):
        DecodedLog(metadata).value("native_tables")


def test_scalar_truncated_or_invalid_value_cannot_be_absence(tmp_path):
    body = (FIXTURES / "tf_table.log").read_text(encoding="utf-8")
    body = body.replace("6.666666666666666e-01", "6.66junk", 1)
    assert decode(tmp_path, text=body)["native_tables"]["status"] == "error"


def test_crlf_and_duplicate_sections_are_not_joined(tmp_path):
    body = (FIXTURES / "disto_harm_table.log").read_text(encoding="utf-8")
    start, end = (
        body.index("                             diode native harmonics"),
        body.index("\f"),
    )
    repeated = body[start:end]
    body = body[:start] + repeated + "\f\n" + body[start:]
    blocks = DecodedLog(decode(tmp_path, text=body.replace("\n", "\r\n"))).value("native_tables")
    assert len(blocks) == 5
    assert blocks[0]["column"] == blocks[1]["column"]
    assert [r["frequency"] for r in blocks[0]["rows"]] == [
        r["frequency"] for r in blocks[1]["rows"]
    ]
    assert blocks[0]["ordinal"] != blocks[1]["ordinal"]


def test_real_frequency_syntax_is_a_synthetic_format_case(tmp_path):
    body = (FIXTURES / "disto_harm_table.log").read_text(encoding="utf-8")
    body = re.sub(r",\t[+-]?\d[^\n]*", "", body)
    blocks = DecodedLog(decode(tmp_path, text=body)).value("native_tables")
    assert all(b["column"]["representation"] == "real" for b in blocks)
    assert all(r["imag"] is None for b in blocks for r in b["rows"])


def test_nonfinite_coordinates_components_and_scalar_are_null_counted(tmp_path):
    body = (
        (FIXTURES / "pz_table.log")
        .read_text(encoding="utf-8")
        .replace("-1.00000000000000e+03,0.000000000000000e+00", "nan, inf")
    )
    metadata = decode(tmp_path, text=body)
    section = DecodedLog(metadata).section("native_tables")
    assert section["nonfinite_count"] == 2
    assert section["value"][0]["entries"][0]["real"] is None
    root = tmp_path / "frequency"
    root.mkdir()
    body = (
        (FIXTURES / "disto_harm_table.log")
        .read_text(encoding="utf-8")
        .replace(
            "0\t1.000000000000000e+02\t0.000000000000000e+00,\t0.000000000000000e+00",
            "0 nan inf, -nan",
            1,
        )
    )
    section = DecodedLog(decode(root, text=body)).section("native_tables")
    assert section["nonfinite_count"] == 3
    assert section["value"][0]["rows"][0] == {
        "line": 24,
        "index": 0,
        "frequency": None,
        "real": None,
        "imag": None,
    }


def test_native_caps_refuse_before_numeric_row_parsing(tmp_path, monkeypatch):
    from ltspice_mcp.lib import native_log_tables

    def forbidden(*_args):
        pytest.fail("Numeric rows parsed after exhausted accumulation budget")

    monkeypatch.setattr(native_log_tables, "_number", forbidden)
    with pytest.raises(LogLimitError, match="section_entries"):
        decode(tmp_path, case="tf_table", limits=replace(LIMITS, section_entries=15))


def test_status_or_model_assignments_outside_native_context_are_absent(tmp_path):
    metadata = decode(tmp_path, text="Circuit: test\ntemp=27\nmodel=2\n", console="gain = 3\n")
    assert metadata["native_tables"] == {
        "status": "absent",
        "value": [],
        "error": None,
        "nonfinite_count": 0,
    }


def test_parent_strictly_requires_eighth_field(tmp_path):
    metadata = decode(tmp_path, case="tf_table")
    metadata.pop("native_tables")
    with pytest.raises(ValueError, match="native_tables"):
        DecodedLog(metadata)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("plot_id",), "tf1"),
        (("step_index",), 0),
        (("ordinal",), 1),
        (("line_start",), 0),
        (("line_end",), 1),
        (("closed_by",), "eof"),
        (("analysis_extent",), "complete"),
        (("entries", 0, "unit"), "V"),
        (("entries", 0, "real"), True),
        (("entries", 0, "line"), 99),
        (("entries", 0, "imag"), 2.0),
    ],
)
def test_parent_rejects_invented_identity_units_closure_or_bad_numbers(tmp_path, path, value):
    metadata = decode(tmp_path, case="tf_table")
    block = metadata["native_tables"]["value"][0]
    target = block
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=r".+"):
        DecodedLog(metadata)


def test_native_accessors_are_copy_isolated(tmp_path):
    metadata = decode(tmp_path, case="tf_table")
    expected = copy.deepcopy(metadata)
    log = DecodedLog(metadata)
    metadata["native_tables"]["value"][0]["entries"].clear()
    log.value("native_tables")[0]["entries"][0]["real"] = 99
    assert log.as_dict() == expected
