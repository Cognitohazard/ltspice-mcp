"""Background work has an owner that can say when it has settled.

A task started and forgotten leaves anything that needs its effect polling
for a side effect or sleeping. These pin what ``settled`` promises: it waits
for the work its owner started, including work started while it waits, and
for nothing else.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ltspice_mcp.lib import recent
from ltspice_mcp.lib.background import BackgroundTasks
from ltspice_mcp.server import _notice_circuit
from ltspice_mcp.state import SessionState


async def test_settled_waits_for_work_started_while_it_waits():
    owner = BackgroundTasks()
    release = asyncio.Event()
    finished: list[str] = []

    async def second() -> None:
        finished.append("second")

    async def first() -> None:
        await release.wait()
        owner.spawn(second())
        finished.append("first")

    owner.spawn(first())
    settling = asyncio.ensure_future(owner.settled())
    await asyncio.sleep(0)
    assert not settling.done()

    release.set()
    await settling

    assert finished == ["first", "second"]
    assert not owner


async def test_settled_for_a_key_does_not_wait_for_other_keys():
    owner = BackgroundTasks()
    held = asyncio.Event()

    async def done() -> None:
        return None

    other = owner.spawn(held.wait(), key="job-b")
    owner.spawn(done(), key="job-a")

    await owner.settled("job-a")

    assert owner.pending() == [other]
    held.set()
    await owner.settled()


async def test_a_failed_task_settles_without_raising_here():
    owner = BackgroundTasks()

    async def fail() -> None:
        raise RuntimeError("the task's own failure")

    task = owner.spawn(fail())
    await owner.settled()

    assert not owner
    with pytest.raises(RuntimeError, match="own failure"):
        task.result()


async def test_session_settles_after_its_recent_index_write(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A tool call naming a circuit records it in the recent index without
    waiting for the write; the session's ``settled`` is what waits for it."""
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "home"))
    circuit = work_dir / "rc.cir"
    circuit.write_text("* rc\nR1 in out 1k\n.end\n")

    await _notice_circuit({"path": str(circuit)}, state_no_sim)
    await state_no_sim.settled()

    assert [entry["path"] for entry in recent.load()] == [str(circuit.resolve())]
