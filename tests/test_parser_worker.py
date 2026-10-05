"""Recorded artifacts through the real contained decoder and plain manifest."""

import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import pytest

from ltspice_mcp.lib.parser_process import ParserProcessLimits, run_parser_sync
from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
from ltspice_mcp.lib.store import Store
from tests.conftest import LIVENESS_S
from tests.test_raw_header import LIMITS

FIXTURES = Path(__file__).parent / "fixtures"
PROCESS_LIMITS = ParserProcessLimits(2 * 1024**3, 65536, 2 * 1024**2, 65536, 3)


def request_for(raw):
    return {
        "version": 1,
        "op": "load_raw",
        "sources": {
            "raw": str(raw),
            "log": str(raw.with_suffix(".log")),
            "console": str(raw.with_suffix(".exe.log")),
        },
        "dialect": None,
        "producing_dialect": None,
        "limits": {
            "raw": asdict(LIMITS),
            "log": {
                "log_bytes": 1024**2,
                "line_bytes": 65536,
                "lines": 20000,
                "section_entries": 50000,
                "metadata_bytes": 2 * 1024**2,
            },
            "step_rows": 1000,
            "metadata_bytes": 2 * 1024**2,
        },
        "existing_cache_keys": [],
    }


@pytest.mark.parametrize(
    ("name", "plots"),
    [
        ("ltspice_tran_rc.raw", 1),
        ("ltspice_step_ac.raw", 1),
        ("ngspice_noise_2plot.raw", 2),
    ],
)
def test_real_worker_returns_complete_resident_plots_after_reaping(tmp_path, name, plots):
    directory = Store(tmp_path).parser_dir("captured")
    directory.mkdir(parents=True)
    request = request_for((FIXTURES / name).resolve())
    reply = run_parser_sync(
        request,
        work_dir=directory,
        deadline=time.monotonic() + LIVENESS_S,
        limits=PROCESS_LIMITS,
    )
    assert not psutil.pid_exists(reply.worker_pid)
    parsed = read_parsed_artifacts(
        reply.metadata, directory, limits=LIMITS, require_raw=True, request=request
    )
    raw = parsed.raw
    assert raw is not None
    assert len(raw.plots) == plots
    arrays = [plot.get_wave(plot.get_trace_names()[-1]) for plot in raw.plots]
    expected = [wave.copy() for wave in arrays]
    shutil.rmtree(directory)
    for wave, copy in zip(arrays, expected, strict=True):
        assert not wave.flags.writeable
        np.testing.assert_array_equal(wave, copy)


def test_cache_hit_recaptures_bytes_and_companion_presence(tmp_path):
    raw_path = tmp_path / "circuit.raw"
    raw_path.write_bytes((FIXTURES / "ltspice_tran_rc.raw").read_bytes())
    request = request_for(raw_path)

    def call(name):
        directory = Store(tmp_path).parser_dir(name)
        directory.mkdir(parents=True)
        reply = run_parser_sync(
            request,
            work_dir=directory,
            deadline=time.monotonic() + LIVENESS_S,
            limits=PROCESS_LIMITS,
        )
        assert not psutil.pid_exists(reply.worker_pid)
        return reply.metadata

    first = call("first")
    assert first["status"] == "ok"
    request["existing_cache_keys"] = [first["cache_key"]]
    second = call("second")
    assert second == {"version": 1, "status": "cached", "cache_key": first["cache_key"]}
    raw_path.with_suffix(".log").write_text("No step metadata\n", encoding="utf-8")
    third = call("third")
    assert third["status"] == "ok"
    assert third["cache_key"] != first["cache_key"]


@pytest.mark.parametrize("raw_kind", ["missing", "malformed", "valid"])
def test_log_operation_captures_facts_without_requiring_raw_decode(tmp_path, raw_kind):
    raw_path = tmp_path / "circuit.raw"
    if raw_kind != "missing":
        raw_path.write_bytes(
            b"not a RAW file"
            if raw_kind == "malformed"
            else (FIXTURES / "ltspice_tran_rc.raw").read_bytes()
        )
    raw_path.with_suffix(".log").write_bytes((FIXTURES / "ltspice_tran_rc.log").read_bytes())
    request = request_for(raw_path)
    request["op"] = "load_logs"
    directory = Store(tmp_path).parser_dir("logs")
    directory.mkdir(parents=True)
    reply = run_parser_sync(
        request,
        work_dir=directory,
        deadline=time.monotonic() + LIVENESS_S,
        limits=PROCESS_LIMITS,
    )
    assert not psutil.pid_exists(reply.worker_pid)
    parsed = read_parsed_artifacts(
        reply.metadata,
        directory,
        limits=LIMITS,
        require_raw=False,
        request=request,
    )
    assert parsed.raw is None
    shutil.rmtree(directory)
    measurements = parsed.logs.section("measurements")
    assert measurements["status"] == "parsed"
    assert measurements["value"]["measurements"]["vfinal"]["values"] == [0.999876166042]
    assert ("raw" in parsed.logs.captured.absent) == (raw_kind == "missing")


def test_raw_and_log_operations_share_captured_identity(tmp_path):
    request = request_for((FIXTURES / "ltspice_tran_rc.raw").resolve())
    first_dir = Store(tmp_path).parser_dir("raw")
    first_dir.mkdir(parents=True)
    first = run_parser_sync(
        request,
        work_dir=first_dir,
        deadline=time.monotonic() + LIVENESS_S,
        limits=PROCESS_LIMITS,
    )
    request["op"] = "load_logs"
    request["existing_cache_keys"] = [first.metadata["cache_key"]]
    second_dir = Store(tmp_path).parser_dir("logs")
    second_dir.mkdir()
    second = run_parser_sync(
        request,
        work_dir=second_dir,
        deadline=time.monotonic() + LIVENESS_S,
        limits=PROCESS_LIMITS,
    )
    assert second.metadata == {
        "version": 1,
        "status": "cached",
        "cache_key": first.metadata["cache_key"],
    }
