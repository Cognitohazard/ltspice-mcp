"""Event-loop responsiveness: heavy parses must not stall concurrent requests.

The MCP SDK dispatches every incoming request as its own asyncio task on one
shared event loop, so a handler that blocks the loop (e.g. a multi-second
RawRead parse of a large ``.raw``) freezes every other in-flight request —
including ``cancel_job`` — and even the transport's receive loop, until it
returns. These tests drive a deliberately slow parse and a light tool
concurrently through their real handler entry points and assert the light
request is served while the heavy one is still in flight.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import AnyUrl
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.lib import recent, services
from ltspice_mcp.server import read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import SignalStatsInput, handle_signal_stats
from ltspice_mcp.tools.status import ServerStatusInput, handle_server_status
from tests.conftest import _FakeServer, make_sim_job, stage_recorded_fixture

# Stands in for a multi-hundred-MB parse over /mnt/c. The only deliberate
# slow-op in this module; every timing assertion keeps >=4x margin to it.
SLOW_OP_SECONDS = 1.0


def slow_rawread(*args, **kwargs):
    """RawRead stand-in that blocks for SLOW_OP_SECONDS before parsing."""
    time.sleep(SLOW_OP_SECONDS)
    return RawRead(*args, **kwargs)


async def assert_light_request_served(heavy: asyncio.Task, state: SessionState) -> None:
    """Serve ``server_status`` while ``heavy`` is in flight; assert it
    returns promptly and before the heavy task completes."""
    t0 = time.monotonic()
    light = await handle_server_status(ServerStatusInput(), state)
    light_elapsed = time.monotonic() - t0

    assert not heavy.done(), (
        "heavy operation finished before the light request was even served — "
        "it ran inline on the event loop and stalled all other requests"
    )
    # The light handler is ms-scale; half the slow-op duration is a generous
    # bound that still proves it was not queued behind the full parse.
    assert light_elapsed < SLOW_OP_SECONDS / 2
    assert light.content


async def test_light_tool_served_while_heavy_parse_in_flight(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A slow RawRead parse must not block a concurrent light request.

    Drives ``signal_stats`` (heavy: parses a recorded LTspice AC raw through
    services.load_raw, with the parse patched to take SLOW_OP_SECONDS) and
    ``server_status`` (light: no file I/O) concurrently on one event loop.
    """
    raw_path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    monkeypatch.setattr(services, "RawRead", slow_rawread)

    heavy = asyncio.create_task(
        handle_signal_stats(
            SignalStatsInput(raw_file=str(raw_path), signal="V(out)"),
            state_no_sim,
        )
    )
    # One loop tick: the heavy handler starts and reaches the parse.
    await asyncio.sleep(0)

    await assert_light_request_served(heavy, state_no_sim)

    # The offloaded parse must still produce the correct result afterward.
    result = await heavy
    sc = result.structuredContent
    assert sc is not None
    assert sc["analysis_type"] == "ac"
    assert sc["point_count"] == 81  # dec 20 over 4 decades, recorded fixture


async def test_recent_index_write_runs_off_loop(
    state_no_sim: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The recent-circuits write (cross-process lock poll + durable fsync)
    must not stall the loop while it is held up.

    If the write ran inline, the ``await asyncio.sleep`` below could not
    complete until the slow touch returned — so reaching the ``not done``
    assertion at all proves the loop stayed live during the write.
    """
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
    state_no_sim.config.persist_jobs = True
    circuit = tmp_path / "rc.cir"
    circuit.write_text("* rc\n.end\n")
    resolved = circuit.resolve()

    real_touch = recent.touch

    def slow_touch(p, **kwargs):
        time.sleep(SLOW_OP_SECONDS)  # stands in for a contended cross-process lock
        real_touch(p, **kwargs)

    monkeypatch.setattr(recent, "touch", slow_touch)

    write = asyncio.create_task(state_no_sim.note_recent_circuit(resolved))
    await asyncio.sleep(0)  # let the write task start and reach the touch

    await asyncio.sleep(0.05)  # a loop tick, far shorter than the slow touch
    assert not write.done(), (
        "recent-index write finished before a 50 ms loop tick — it ran inline on the event loop"
    )

    await write
    entries = recent.load()
    assert [e["path"] for e in entries] == [str(resolved)]


async def test_resource_read_served_off_loop(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A slow RawRead parse inside an MCP resource read must not block a
    concurrent light request.

    Drives the real router seam — ``server.read_resource`` over the
    ``ltspice://results/{job}/signals`` route, with the parse patched to
    take SLOW_OP_SECONDS — concurrently with ``server_status``.
    """
    raw_path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    job = make_sim_job("resjob", raw_file=raw_path)
    state_no_sim.jobs[job.job_id] = job

    monkeypatch.setattr(services, "RawRead", slow_rawread)

    with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
        # Keep this wrapper: create_task needs a true coroutine, and the
        # SDK's read_resource is typed as returning a plain Awaitable.
        async def _read_signals_resource():
            return await read_resource(AnyUrl("ltspice://results/resjob/signals"))

        heavy = asyncio.create_task(_read_signals_resource())
        # One loop tick: the read task starts and hands the router to a worker.
        await asyncio.sleep(0)

        await assert_light_request_served(heavy, state_no_sim)

        # The offloaded read must still produce the correct result afterward.
        result = await heavy
        assert not isinstance(result, str | bytes)
        contents = list(result)
        assert len(contents) == 1
        body = contents[0].content
        assert isinstance(body, str)
        assert "V(out)" in body
