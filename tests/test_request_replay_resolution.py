"""A request replay finds its job as every other read of a job id does.

``run_experiments(request_id=...)`` replaying a recorded submission resolves
the job through the registry: a record whose owner exited is reconciled and
written back, exactly as ``jobs`` or ``analyze_results`` would leave it, and a
replay is not logged as a new submission. Reading a record straight from the
store is banned outside the registry and its peers in ``pyproject.toml``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import handle_run_experiments
from tests._source_scan import PACKAGE, parsed
from tests.conftest import FakeSim, await_until, exited_pid, fake_simulator
from tests.test_run_experiments import _args, _assert_schema, _deck


@pytest.mark.asyncio
async def test_a_replay_of_an_exited_owner_s_job_writes_back_its_reconciliation(
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """The coordinator runs here, but every record it writes names a process
    that has exited, so to a second session the job's owner is gone."""
    exited = exited_pid()
    save_job = experiment_store.save_job
    monkeypatch.setattr(
        experiment_store,
        "save_job",
        lambda job: save_job(dataclasses.replace(job, owner_pid=exited)),
    )
    held: list[Callable[[], None]] = []
    fake_simulator(monkeypatch, delay_s=None, held=held)
    deck = _deck(work_dir / "exited-replay.cir")
    first = _assert_schema(
        await handle_run_experiments(_args(deck, "exited-replay", wait_s=0), state_with_sim)
    )
    await await_until(lambda: len(held) == 1, what="the case to reach the simulator")
    await state_with_sim.job_registry.drain_pending()
    record = Store(work_dir).job_record(first["job_id"])
    assert json.loads(record.read_text(encoding="utf-8"))["status"] in {"queued", "running"}
    other = SessionState.create(state_with_sim.config, available={"fake": FakeSim})
    caplog.set_level(logging.INFO, logger="ltspice_mcp.events")
    try:
        replayed = _assert_schema(
            await handle_run_experiments(_args(deck, "exited-replay", wait_s=0), other)
        )
        await other.job_registry.drain_pending()

        assert replayed["replayed"] is True
        assert replayed["status"] == "interrupted"
        assert json.loads(record.read_text(encoding="utf-8"))["status"] == "interrupted"
        events = [
            getattr(entry, "ltspice_event", {}).get("event")
            for entry in caplog.records
            if entry.name == "ltspice_mcp.events"
        ]
        assert "submitted" not in events
    finally:
        held[0]()
        await state_with_sim.settled()


# ---------------------------------------------------------------------------
# One route from a job id to a job
# ---------------------------------------------------------------------------

#: Who may register a job: the runner that admits it, the registry, and the
#: session's delegate to the registry.
_REGISTRARS = frozenset({"lib/experiment_runner.py", "lib/job_registry.py", "state.py"})


def _named_uses(names: frozenset[str]) -> list[tuple[str, int]]:
    """Every call of, or reference to, one of ``names``: a method passed on as a
    callback is used as surely as one called in place."""
    found: list[tuple[str, int]] = []
    for path in PACKAGE.rglob("*.py"):
        module = path.relative_to(PACKAGE).as_posix()
        for node in ast.walk(parsed(path)):
            if (isinstance(node, ast.Attribute) and node.attr in names) or (
                isinstance(node, ast.Name) and node.id in names
            ):
                found.append((module, node.lineno))
    return found


def test_only_the_runner_and_the_registry_register_a_job() -> None:
    stray = sorted(
        f"{module}:{line}"
        for module, line in _named_uses(frozenset({"add_experiment_job"}))
        if module not in _REGISTRARS
    )
    assert not stray, f"jobs registered around the runner and registry: {stray}"
