"""Malformed worker output must fail before it becomes an analysis array."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from ltspice_mcp.lib.parser_protocol import read_decoded_raw
from ltspice_mcp.lib.raw_header import preflight_raw
from tests.test_raw_header import LIMITS, synthetic_binary


@pytest.fixture
def parsed_manifest(tmp_path):
    raw = tmp_path / "input.raw"
    raw.write_bytes(synthetic_binary())
    header = preflight_raw(raw, limits=LIMITS)
    arrays = []
    for index, values in enumerate(([0.0, 1.0], [1.0, 2.0])):
        data = np.asarray(values, dtype="<f8").tobytes()
        name = f"p0_t{index}.bin"
        (tmp_path / name).write_bytes(data)
        arrays.append(
            {
                "plot_index": 0,
                "trace_index": index,
                "file": name,
                "dtype": "<f8",
                "count": 2,
                "byte_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "version": 1,
        "status": "ok",
        "cache_key": "c" * 64,
        "header": asdict(header),
        "arrays": arrays,
        "plots": [
            {
                "plot_index": 0,
                "decoder_has_axis": True,
                "data_convention": "stored",
                "step_status": "unstepped",
                "step_ranges": [{"step_index": 0, "offset": 0, "length": 2, "log_row": None}],
            }
        ],
        "step_log": {
            "present": False,
            "size_bytes": None,
            "sha256": None,
            "encoding": None,
            "status": "absent",
            "rows": [],
        },
    }


def test_manifest_loads_owned_readonly_arrays(parsed_manifest, tmp_path):
    result = read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)
    np.testing.assert_array_equal(result.get_axis(), [0, 1])
    np.testing.assert_array_equal(result.get_wave("V(out)"), [1, 2])
    assert not result.get_wave("V(out)").flags.writeable
    assert result.descriptor.snapshot_id == "c" * 64
    for path in tmp_path.iterdir():
        path.unlink()
    np.testing.assert_array_equal(result.get_wave("V(out)"), [1, 2])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "O"),
        ("count", 10**12),
        ("byte_size", 1),
        ("file", "../outside.bin"),
        ("trace_index", 1),
    ],
)
def test_bad_array_manifest_refuses_before_allocation(
    parsed_manifest, tmp_path, monkeypatch, field, value
):
    parsed_manifest["arrays"][0][field] = value

    def refuse_read(*args, **kwargs):
        pytest.fail("Malformed numeric declarations must be refused before allocation")

    monkeypatch.setattr(np, "fromfile", refuse_read)
    with pytest.raises(ValueError, match=r"(validation error|Parser)"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)


@pytest.mark.parametrize("change", ["truncated", "extended", "different"])
def test_numeric_bytes_must_match_manifest(parsed_manifest, tmp_path, change):
    path = tmp_path / "p0_t1.bin"
    data = path.read_bytes()
    if change == "truncated":
        data = data[:-1]
    elif change == "extended":
        data += b"extra"
    else:
        data = bytes([data[0] ^ 1]) + data[1:]
    path.write_bytes(data)
    with pytest.raises(ValueError, match=r"(size|digest) disagrees"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)


@pytest.mark.parametrize(("field", "value"), [("offset", 1), ("length", 1), ("log_row", 0)])
def test_step_ranges_cannot_invent_or_drop_points(parsed_manifest, tmp_path, field, value):
    parsed_manifest["plots"][0]["step_ranges"][0][field] = value
    with pytest.raises(ValueError, match="Step"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)


@pytest.mark.parametrize(
    "change", ["missing_digest", "extra_row", "wrong_ordinal", "unstepped_binding", "plot_bytes"]
)
def test_inconsistent_metadata_refuses_before_allocation(
    parsed_manifest, tmp_path, monkeypatch, change
):
    parsed_manifest["step_log"] = {
        "present": True,
        "size_bytes": 12,
        "sha256": "a" * 64,
        "encoding": "utf-8",
        "status": "parsed",
        "rows": [
            {
                "ordinal": 0,
                "line_number": 1,
                "text": ".step r=1",
                "parameters": [{"name": "r", "token": "1", "value": 1.0}],
            }
        ],
    }
    plot = parsed_manifest["plots"][0]
    plot["step_status"] = "matched"
    plot["step_ranges"][0]["log_row"] = 0
    if change == "missing_digest":
        parsed_manifest["step_log"]["sha256"] = None
    elif change == "extra_row":
        parsed_manifest["step_log"]["rows"].append(
            {
                "ordinal": 1,
                "line_number": 2,
                "text": ".step r=2",
                "parameters": [{"name": "r", "token": "2", "value": 2.0}],
            }
        )
    elif change == "wrong_ordinal":
        parsed_manifest["step_log"]["rows"][0]["ordinal"] = 2
    elif change == "unstepped_binding":
        plot["step_status"] = "unstepped"
    else:
        parsed_manifest["header"]["plots"][0]["numeric_bytes"] = 1

    def refuse_read(*args, **kwargs):
        pytest.fail("Inconsistent metadata must refuse before numeric allocation")

    monkeypatch.setattr(np, "fromfile", refuse_read)
    with pytest.raises(ValueError, match=r"(Step|Parser|Present)"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)


@pytest.mark.parametrize("change", ["step_header", "later_file"])
def test_all_metadata_and_file_sizes_are_admitted_before_arrays(
    parsed_manifest, tmp_path, monkeypatch, change
):
    if change == "step_header":
        parsed_manifest["header"]["plots"][0]["flags"] = ("real", "stepped")
    else:
        with (tmp_path / "p0_t1.bin").open("ab") as handle:
            handle.write(b"extra")
    reads = []
    original = np.fromfile

    def observed(*args, **kwargs):
        reads.append(kwargs.get("count"))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", observed)
    with pytest.raises(ValueError, match=r"(Unstepped status|file size)"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)
    assert reads == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("status", "cached"),
        ("cache_key", "f" * 64),
    ],
)
def test_nested_raw_cannot_override_shared_artifact_envelope(
    parsed_manifest, tmp_path, monkeypatch, field, value
):
    from ltspice_mcp.lib.log_decode import LogLimits, decode_logs
    from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs, parser_cache_key
    from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
    from tests.test_parser_worker import request_for

    raw_path = tmp_path / "input.raw"
    request = request_for(raw_path)
    directory = tmp_path / "captured"
    directory.mkdir()
    captured = capture_inputs(
        SourceFiles(raw=raw_path, log=None, console=None),
        directory,
        input_bytes=LIMITS.input_bytes,
        log_bytes=request["limits"]["log"]["log_bytes"],
    )
    logs = decode_logs(captured, directory, limits=LogLimits(**request["limits"]["log"]))
    key = parser_cache_key(
        captured,
        dialect=None,
        producing_dialect=None,
        limits=request["limits"],
    )
    raw = {
        key: value
        for key, value in parsed_manifest.items()
        if key not in {"version", "status", "cache_key"}
    }
    raw[field] = value
    metadata = {"version": 1, "status": "ok", "cache_key": key, "raw": raw, "logs": logs}
    reads = []
    original = np.fromfile

    def observed(*args, **kwargs):
        reads.append(kwargs.get("count"))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", observed)
    with pytest.raises(ValueError, match="Extra inputs"):
        read_parsed_artifacts(
            metadata,
            tmp_path,
            limits=LIMITS,
            require_raw=True,
            request=request,
        )
    assert reads == []


def _worker_reply(tmp_path, source):
    from ltspice_mcp.lib.parser_worker import parse_request
    from tests.test_parser_worker import request_for

    request = request_for(source.resolve())
    directory = tmp_path / "worker"
    directory.mkdir()
    parse_request(request, directory)
    return json.loads((directory / "result.json").read_bytes()), directory, request


@pytest.mark.parametrize("offset", ["nan", "not-a-number"])
def test_invalid_offset_is_refused_before_resident_reads(tmp_path, monkeypatch, offset):
    from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts

    source = tmp_path / "offset.raw"
    source.write_bytes(
        synthetic_binary(
            plot="Transient Analysis",
            flags="real double",
            variables=(("time", "time"), ("V(out)", "voltage")),
            command="Linear Technology Corporation LTspice XVII",
            extra=f"Offset: {offset}\n",
        )
    )
    result, directory, request = _worker_reply(tmp_path, source)
    reads = []
    original = np.fromfile

    def observed(*args, **kwargs):
        reads.append(kwargs.get("count"))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", observed)
    with pytest.raises(
        ValueError, match=r"(offset must be finite|could not convert string to float)"
    ):
        read_parsed_artifacts(result, directory, limits=LIMITS, require_raw=True, request=request)
    assert reads == []


def test_raw_step_values_must_agree_with_captured_log_facts(tmp_path, monkeypatch):
    from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
    from tests.test_parser_worker import FIXTURES

    result, directory, request = _worker_reply(tmp_path, FIXTURES / "ltspice_step_ac.raw")
    assert result["logs"]["steps"]["value"][0] == {"r": 1000.0}
    # Corrupt the worker reply while preserving the original token and capture.
    result["raw"]["step_log"]["rows"][0]["parameters"][0]["value"] = 9999.0
    reads = []
    original = np.fromfile

    def observed(*args, **kwargs):
        reads.append(kwargs.get("count"))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", observed)
    with pytest.raises(ValueError, match="Step values"):
        read_parsed_artifacts(result, directory, limits=LIMITS, require_raw=True, request=request)
    assert reads == []


def test_stored_step_parameters_must_be_the_plots_param_variables(parsed_manifest, tmp_path):
    """A worker claiming a plot stores its own step values is held to the
    header: the variables it names must be the plot's leading ``param`` ones,
    which this plot's time axis is not."""
    plot = parsed_manifest["plots"][0]
    plot["step_status"] = "matched"
    plot["step_ranges"] = [
        {"step_index": index, "offset": index, "length": 1, "log_row": None} for index in range(2)
    ]
    plot["step_parameters"] = [0]
    with pytest.raises(ValueError, match="Stored step parameters"):
        read_decoded_raw(parsed_manifest, tmp_path, limits=LIMITS)
