"""Event-loop responsiveness: heavy parses must not stall concurrent requests.

The MCP SDK dispatches every incoming request as its own asyncio task on one
shared event loop, so a handler that blocks the loop (e.g. a multi-second
RawRead parse of a large ``.raw``) freezes every other in-flight request —
including ``cancel_job`` — and even the transport's receive loop, until it
returns. These tests drive a held parse and a light tool concurrently through
their real handler entry points and assert the light request is served while
the heavy one is still in flight.

The heavy work is held, not slowed: it signals when it starts and waits until
the test releases it, so "still in flight" is a fact the test arranged rather
than a race against a fixed duration.
"""

import asyncio
import threading
from pathlib import Path

import pytest
from mcp import types

from ltspice_mcp import resources
from ltspice_mcp.lib import parser_service, recent, services
from ltspice_mcp.lib.metrics import signal_stats
from ltspice_mcp.lib.recipes import SignalStatsRecipe
from ltspice_mcp.server import read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests.conftest import LIVENESS_S, await_until, fake_request_context, stage_recorded_fixture


class _Held:
    """Stands in for a multi-hundred-MB parse over /mnt/c: blocking work that
    says when it has started and runs until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def hold(self) -> None:
        self.entered.set()
        # Bounded, so work that regressed onto the event loop (where the test
        # can never get to release it) fails the test instead of hanging it.
        self.release.wait(LIVENESS_S)


async def assert_light_request_served(heavy: asyncio.Task, state: SessionState) -> None:
    """Serve an ``inspect`` capabilities query (a registered consolidated
    tool with no file I/O) while ``heavy`` is held in flight.

    Work run inline on the loop would block it until the hold timed out, and
    the heavy task would then finish before the light request was served.
    """
    light = await handle_inspect(
        InspectInput.model_validate({"queries": [{"kind": "capabilities"}]}), state
    )
    assert not heavy.done(), (
        "heavy operation finished before the light request was even served — "
        "it ran inline on the event loop and stalled all other requests"
    )
    assert light.content


async def test_light_tool_served_while_heavy_parse_in_flight(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A held parser supervisor must not block a concurrent light request.

    Drives ``signal_stats`` (heavy: parses a recorded LTspice AC raw through
    services.load_raw, with supervision held) and an ``inspect`` capabilities
    query (light: no file I/O) concurrently on one event loop.
    """
    raw_path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    run = parser_service.run_parser_sync
    held = _Held()

    def held_parser(*args, **kwargs):
        held.hold()
        return run(*args, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", held_parser)

    heavy = asyncio.create_task(
        signal_stats(
            services.AnalysisSource.for_raw(raw_path),
            SignalStatsRecipe(key="s", metric="signal_stats", signal="V(out)"),
            0,
            state_no_sim,
        )
    )
    try:
        await await_until(held.entered.is_set)
        await assert_light_request_served(heavy, state_no_sim)
    finally:
        held.release.set()

    # The offloaded parse must still produce the correct result afterward.
    sc = await heavy
    assert sc["analysis_type"] == "ac"
    assert sc["point_count"] == 81  # dec 20 over 4 decades, recorded fixture


async def test_recent_index_write_runs_off_loop(
    state_no_sim: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The recent-circuits write (cross-process lock poll + durable fsync)
    must not stall the loop while it is held up.

    If the write ran inline, the loop could not run the test again until the
    held touch timed out, by which point the write would have finished.
    """
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
    state_no_sim.config.persist_jobs = True
    circuit = tmp_path / "rc.cir"
    circuit.write_text("* rc\n.end\n")
    resolved = circuit.resolve()

    real_touch = recent.touch
    held = _Held()

    def held_touch(p, **kwargs):
        held.hold()  # stands in for a contended cross-process lock
        real_touch(p, **kwargs)

    monkeypatch.setattr(recent, "touch", held_touch)

    write = asyncio.create_task(state_no_sim.note_recent_circuit(resolved))
    try:
        await await_until(held.entered.is_set)
        assert not write.done(), "recent-index write finished while held — it ran inline"
    finally:
        held.release.set()

    await write
    entries = recent.load()
    assert [e["path"] for e in entries] == [str(resolved)]


async def test_resource_read_served_off_loop(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A slow netlist decode inside an MCP resource read must not block a
    concurrent light request.

    Drives the real router seam — ``server.read_resource`` over the
    ``spice://netlists/{filename}`` route, with the decode held — concurrently
    with a light ``inspect`` query.
    """
    deck = work_dir / "slow.cir"
    deck.write_text("* slow read\nR1 in 0 1k\n.end\n", encoding="utf-8")

    real_read = resources.read_spice_text
    held = _Held()

    def held_read(path):
        held.hold()
        return real_read(path)

    monkeypatch.setattr(resources, "read_spice_text", held_read)

    heavy = asyncio.create_task(
        read_resource(
            fake_request_context(state_no_sim),
            types.ReadResourceRequestParams(uri="spice://netlists/slow.cir"),
        )
    )
    try:
        # The decode has started in its worker, so the read is in flight.
        await await_until(held.entered.is_set)
        await assert_light_request_served(heavy, state_no_sim)
    finally:
        held.release.set()

    # The offloaded read must still produce the correct result afterward.
    result = await heavy
    assert len(result.contents) == 1
    entry = result.contents[0]
    assert isinstance(entry, types.TextResourceContents)
    assert "R1 in 0 1k" in entry.text
