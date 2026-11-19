"""One bound on the whole call, and a repeat that leaves its artifacts alone.

``[analysis] analysis_budget_s`` bounds a whole ``analyze_results`` call: the
read of the sources, every recipe, the check after the reads and the signal
listing. A recipe the bound leaves no time for fails alone, and says to ask
for it in a call of its own. A repeat of a request writes no artifact it
already published, so a file another program holds open (a CSV in a
spreadsheet, on Windows) does not fail it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded
from ltspice_mcp.lib import atomic_write
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import LIVENESS_S, SyncApi, stage_recorded_fixture
from tests.test_analyze_results_through_the_presenter import _analyze

_CSV = {"key": "csv", "metric": "waveform", "signals": ["V(out)"], "format": "csv"}


def _slow_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CSV writer that holds until its deadline fires: the real abort
    callback the analysis hands it, polled."""
    from ltspice_mcp.tools import analysis

    def slow_writer(*args: Any) -> None:
        out_path, should_abort = args[8], args[9]
        with atomic_write(out_path) as handle:
            handle.write("partial\n")
            started = time.monotonic()
            while not should_abort():
                # timing: a fake writer spinning until its abort callback fires
                time.sleep(0.002)
                assert time.monotonic() - started < LIVENESS_S
            raise AnalysisDeadlineExceeded("CSV artifact exceeded its analysis item deadline")

    monkeypatch.setattr(analysis, "build_waveform_csv", slow_writer)
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", lambda recipe, runs: 0.0)


def _request(raw: Path) -> dict[str, Any]:
    return {
        "sources": [{"path": str(raw), "label": "dut"}],
        "recipes": [
            {"key": "first", "metric": "value", "expr": "V(out)", "at": "500u"},
            _CSV,
            {"key": "third", "metric": "value", "expr": "V(out)", "at": "900u"},
        ],
    }


def _bounded_by_the_call(data: dict[str, Any]) -> None:
    assert data["results"]["first"]["values"], "a recipe evaluated in time is answered"
    # The recipe the bound ended fails at its run; the one it left fails alone.
    assert {
        (failure["code"], failure["stage"], failure["where"]) for failure in data["failures"]
    } == {("analysis_deadline", "analyze", "dut:0"), ("analysis_deadline", "deadline", "third")}
    third = next(failure for failure in data["failures"] if failure["where"] == "third")
    assert "call of its own" in third["message"]
    assert not data["results"]["csv"].get("values")
    assert not data["results"]["third"].get("values")


@pytest.mark.asyncio
async def test_one_bound_ends_the_whole_call_and_names_the_recipes_it_left(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    state_no_sim.config.analysis_budget_s = 1.0
    _slow_csv(monkeypatch)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    started = time.monotonic()
    data = await _analyze(state_no_sim, _request(raw))
    # The call's bound, with what the check after the reads may take beyond it.
    assert time.monotonic() - started < 1.0 + LIVENESS_S
    _bounded_by_the_call(data)


def test_the_python_api_has_the_same_bound_and_answer(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    state_no_sim.config.analysis_budget_s = 1.0
    _slow_csv(monkeypatch)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = SyncApi(state_no_sim).analyze_results(**_request(raw))
    _bounded_by_the_call(data)


@pytest.mark.asyncio
async def test_a_repeat_does_not_write_an_artifact_it_already_published(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [_CSV]}
    first = await _analyze(state_no_sim, request)
    (row,) = first["results"]["csv"]["values"]
    path = Path(row["value"]["artifact"]["path"])
    before = await asyncio.to_thread(os.stat, path)
    again = await _analyze(state_no_sim, request)
    (repeat,) = again["results"]["csv"]["values"]
    after = await asyncio.to_thread(os.stat, path)
    assert repeat["value"]["artifact"] == row["value"]["artifact"]
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not list(path.parent.glob("*.pending"))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="a handle that refuses a rename is Windows'")
async def test_a_repeat_succeeds_while_another_program_holds_its_artifact_open(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [_CSV]}
    first = await _analyze(state_no_sim, request)
    path = Path(first["results"]["csv"]["values"][0]["value"]["artifact"]["path"])
    with open(path, "rb"):  # noqa: ASYNC230 - the held handle is the point
        again = await _analyze(state_no_sim, request)
    assert "failures" not in again
    assert again["results"]["csv"]["values"][0]["value"]["artifact"]["path"] == str(path)


@pytest.mark.asyncio
async def test_a_repeat_replaces_a_published_artifact_whose_bytes_differ(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [_CSV]}
    first = await _analyze(state_no_sim, request)
    artifact = first["results"]["csv"]["values"][0]["value"]["artifact"]
    path = Path(artifact["path"])
    whole = await asyncio.to_thread(path.read_bytes)
    await asyncio.to_thread(path.write_bytes, b"x" * len(whole))
    again = await _analyze(state_no_sim, request)
    assert again["results"]["csv"]["values"][0]["value"]["artifact"] == artifact
    assert await asyncio.to_thread(path.read_bytes) == whole
