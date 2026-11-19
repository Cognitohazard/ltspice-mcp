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
from ltspice_mcp.lib import atomic_write, services
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import SyncApi, stage_recorded_fixture
from tests.test_analyze_results_through_the_presenter import _analyze

_CSV = {"key": "csv", "metric": "waveform", "signals": ["V(out)"], "format": "csv"}

#: What the slow writer moves the call's clock by: past any bound a call has.
_PAST_ANY_BOUND_S = 10.0**6
#: A source read longer than the one-second bound these tests once set.
_SLOW_READ_S = 1.5  # timing: fake work, a read slower than the bound these tests once set


class _Delegate:
    """``real``, with the attributes named here replaced."""

    def __init__(self, real: Any, **replaced: Any) -> None:
        self._real = real
        self.__dict__.update(replaced)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _CallClock:
    """The clock ``tools/analyze.py`` reads its bound off (``loop.time()`` and
    ``time.monotonic()``), which a test moves past the bound. The scheduler,
    the parser and every other module keep the real clock, which this one
    never runs behind."""

    def __init__(self) -> None:
        self._spent = 0.0

    def now(self) -> float:
        return time.monotonic() + self._spent

    def spend_the_bound(self) -> None:
        self._spent += _PAST_ANY_BOUND_S


def _call_clock(monkeypatch: pytest.MonkeyPatch) -> _CallClock:
    clock = _CallClock()

    def running_loop() -> Any:
        return _Delegate(asyncio.get_running_loop(), time=clock.now)

    monkeypatch.setattr(analyze_mod, "asyncio", _Delegate(asyncio, get_running_loop=running_loop))
    monkeypatch.setattr(analyze_mod, "time", _Delegate(time, monotonic=clock.now))
    return clock


def _slow_csv(monkeypatch: pytest.MonkeyPatch, *, after_calls: int = 0) -> _CallClock:
    """A CSV writer the call's bound ends, whatever the machine's speed.

    The first ``after_calls`` calls write as the real writer does. The next
    one is the handshake: everything the call did before it is done, and it
    moves the call's clock past any bound, so the abort callback the analysis
    handed it fires and it fails as the real writer does. Every step after it
    finds the bound spent."""
    from ltspice_mcp.tools import analysis

    clock = _call_clock(monkeypatch)
    real = analysis.build_waveform_csv
    calls: list[int] = []

    def slow_writer(*args: Any) -> Any:
        calls.append(1)
        if len(calls) <= after_calls:
            return real(*args)
        out_path, should_abort = args[8], args[9]
        with atomic_write(out_path) as handle:
            handle.write("partial\n")
            clock.spend_the_bound()
            assert should_abort(), "the writer's deadline is the call's bound"
            raise AnalysisDeadlineExceeded("CSV artifact exceeded its analysis item deadline")

    monkeypatch.setattr(analysis, "build_waveform_csv", slow_writer)
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", lambda recipe, runs: 0.0)
    return clock


def _slow_read(monkeypatch: pytest.MonkeyPatch, slow: bool) -> None:
    """With ``slow``, the call's first read of its source, the one that
    starts it, takes longer than the bound these tests once set, as a loaded
    runner's process start can."""
    if not slow:
        return
    real = services.load_artifacts
    reads: list[int] = []

    async def slow_load(*args: Any, **kwargs: Any) -> Any:
        reads.append(1)
        if len(reads) > 1:
            return await real(*args, **kwargs)
        # timing: fake work, a read slower than the one-second bound these tests once set
        await asyncio.sleep(_SLOW_READ_S)
        return await real(*args, **kwargs)

    monkeypatch.setattr(services, "load_artifacts", slow_load)


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
    failures = data.get("failures", [])
    assert data["results"]["first"].get("values"), (
        "a recipe evaluated before the bound is answered",
        failures,
    )
    # The recipe the bound ended fails at its run; the one it left fails alone.
    assert {(failure["code"], failure["stage"], failure["where"]) for failure in failures} == {
        ("analysis_deadline", "analyze", "dut:0"),
        ("analysis_deadline", "deadline", "third"),
    }, failures
    third = next(failure for failure in failures if failure["where"] == "third")
    assert "call of its own" in third["message"]
    assert not data["results"]["csv"].get("values")
    assert not data["results"]["third"].get("values")


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", [False, True], ids=["read", "slow_read"])
async def test_one_bound_ends_the_whole_call_and_names_the_recipes_it_left(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch, slow: bool
):
    _slow_csv(monkeypatch)
    _slow_read(monkeypatch, slow)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(state_no_sim, _request(raw))
    _bounded_by_the_call(data)


@pytest.mark.parametrize("slow", [False, True], ids=["read", "slow_read"])
def test_the_python_api_has_the_same_bound_and_answer(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch, slow: bool
):
    _slow_csv(monkeypatch)
    _slow_read(monkeypatch, slow)
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
