"""A request replay finds its job as every other read of a job id does.

``run_experiments`` repeating a ``request_id`` it has recorded is answered from
the job that id names, on one of three routes: the tool's lookup before it
stages, the request gate a submission enters when that lookup found nothing,
and the recovery journal a recoverable submission is recorded in. A resume
repeating its ``resume_request_id`` replays from the journal the same way. Each
resolves the job through the registry: a record whose owner exited is
reconciled and written back, exactly as ``jobs`` or ``analyze_results`` would
leave it, and a replay is not logged as a new submission. Reading a record
straight from the store is banned outside the registry and its peers in
``pyproject.toml``.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import lookup_root_recovery, resume_experiment
from ltspice_mcp.lib.experiment_runner import (
    ExperimentRunRequest,
    already_staged,
    canonical_fingerprint,
)
from ltspice_mcp.lib.recovery_journal import (
    add_child,
    mark_launched,
    new_root_journal,
    save_journal,
)
from ltspice_mcp.lib.recovery_records import CaseAttempt, ProcessIdentity
from ltspice_mcp.lib.services import resolve_job_async
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import handle_run_experiments
from tests._source_scan import PACKAGE, parsed
from tests.conftest import FakeSim, await_until, exited_pid, fake_simulator
from tests.test_recovery_launch import RecordedNGspice
from tests.test_recovery_launch import committed as committed
from tests.test_run_experiments import _args, _assert_schema, _deck

#: Replays the recorded request from a session that has not seen its job, and
#: says whether the answer was a replay and what status the job had in it.
Replay = Callable[[], Awaitable[tuple[bool, str]]]


@dataclasses.dataclass(frozen=True)
class _Recorded:
    """A submission recorded as queued or running by a process that has exited,
    and its replay from a session that has not read it."""

    record: Path
    session: SessionState
    replay: Replay


@contextlib.asynccontextmanager
async def _submitted(
    route: str, state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Recorded]:
    """A run_experiments submission whose case this process is still running,
    but whose every record names a process that has exited, so to a second
    session the job's owner is gone."""
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
    args = _args(deck, "exited-replay", wait_s=0)
    first = _assert_schema(await handle_run_experiments(args, state))
    await await_until(lambda: len(held) == 1, what="the case to reach the simulator")
    await state.job_registry.drain_pending()
    other = SessionState.create(state.config, available={"fake": FakeSim})

    async def before_staging() -> tuple[bool, str]:
        data = _assert_schema(await handle_run_experiments(args, other))
        return data["replayed"], data["status"]

    async def request_gate() -> tuple[bool, str]:
        # The same request reaching the coordinator, as one does when the
        # lookup before staging ran before the record was written. A replay
        # never stages, so the staging pass is one that refuses to run.
        runner = other.runners.get_experiment_runner(
            asyncio.get_running_loop(),
            FakeSim,
            await asyncio.to_thread(Store(work_dir).runs_root, FakeSim),
            max_parallel=1,
        )
        request = ExperimentRunRequest(
            state=other,
            request_id=args.request_id,
            fingerprint=canonical_fingerprint(args),
            simulator=FakeSim.__name__,
            stage=already_staged,
            simulator_executable=await asyncio.to_thread(executable_identity, FakeSim),
        )
        receipt = await asyncio.shield(runner.submit(request))
        return receipt.replayed, receipt.job.status

    replay = {"before staging": before_staging, "request gate": request_gate}[route]
    try:
        yield _Recorded(Store(work_dir).job_record(first["job_id"]), other, replay)
    finally:
        held[0]()
        await state.settled()


@contextlib.asynccontextmanager
async def _journaled(recoverable: Any) -> AsyncIterator[_Recorded]:
    """A recoverable submission left the same way: only a recoverable submission
    is recorded in a recovery journal, and it replays only from there. Its root
    attempt was launched, so a replay answers from its record rather than
    starting it."""
    _runner, submission, job, store = recoverable
    assert job.recovery is not None
    exited = exited_pid()
    job.status = "running"
    job.owner_pid = exited
    job.recovery = dataclasses.replace(job.recovery, owner=ProcessIdentity(exited, "exited"))
    save_journal(store, mark_launched(new_root_journal(job), job.job_id))
    experiment_store.save_job(job)
    other = SessionState.create(submission.state.config, available={"ngspice": RecordedNGspice})

    async def recovery_journal() -> tuple[bool, str]:
        # What run_experiments asks first for a request_id a journal records.
        receipt = await lookup_root_recovery(
            other,
            request_id=job.request_id,
            fingerprint=job.fingerprint,
            simulator=RecordedNGspice,
            recoverable=True,
        )
        assert receipt is not None
        return receipt.replayed, receipt.job.status

    yield _Recorded(job.store_path, other, recovery_journal)


@contextlib.asynccontextmanager
async def _resumed(recoverable: Any) -> AsyncIterator[_Recorded]:
    """A resume of that submission, its attempt left the same way:
    ``jobs(action="resume")`` repeating its ``resume_request_id`` replays the
    attempt from the journal by the step the journal replay above takes."""
    _runner, submission, parent, store = recoverable
    assert parent.recovery is not None and parent.cases[0].recovery is not None
    # A resume finds the recorded simulator among the session's by class name.
    parent.simulator = RecordedNGspice.__name__
    exited = exited_pid()
    resume_request_id, child_id = "resume-exited", "exp_resumed_0001"
    token = f"{child_id}_case_0"
    child = dataclasses.replace(
        parent,
        job_id=child_id,
        request_id=resume_request_id,
        fingerprint=canonical_fingerprint(
            {
                "parent_job_id": parent.job_id,
                "case_ids": None,
                "retry_failed": False,
                "retry_cancelled": False,
            }
        ),
        store_path=store.job_record(child_id),
        status="running",
        owner_pid=exited,
        recovery=dataclasses.replace(
            parent.recovery,
            parent_job_id=parent.job_id,
            attempt_index=1,
            owner=ProcessIdentity(exited, "exited"),
        ),
        cases=[
            dataclasses.replace(
                parent.cases[0],
                run_token=token,
                recovery=dataclasses.replace(
                    parent.cases[0].recovery, attempt=CaseAttempt(child_id, 1, token)
                ),
            )
        ],
    )
    journal = mark_launched(new_root_journal(parent), parent.job_id)
    journal = add_child(journal, resume_request_id, child.fingerprint, child)
    save_journal(store, mark_launched(journal, child_id))
    experiment_store.save_job(parent)
    experiment_store.save_job(child)
    other = SessionState.create(submission.state.config, available={"ngspice": RecordedNGspice})

    async def resume_replay() -> tuple[bool, str]:
        receipt = await resume_experiment(
            other,
            job_id=parent.job_id,
            resume_request_id=resume_request_id,
            control_token=parent.control_token,
        )
        return receipt.replayed, receipt.job.status

    yield _Recorded(child.store_path, other, resume_replay)


_RECOVERABLE = {"recovery journal": _journaled, "resume replay": _resumed}


@pytest.fixture(params=["before staging", "request gate", *_RECOVERABLE])
def replay_route(request: pytest.FixtureRequest) -> tuple[str, Any]:
    """The route a replay takes, and the session its recorded submission needs."""
    if request.param in _RECOVERABLE:
        return request.param, request.getfixturevalue("committed")
    return request.param, request.getfixturevalue("state_with_sim")


@pytest.mark.asyncio
async def test_a_replay_of_an_exited_owner_s_job_writes_back_its_reconciliation(
    replay_route: tuple[str, Any],
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    route, base = replay_route
    recorded = (
        _RECOVERABLE[route](base)
        if route in _RECOVERABLE
        else _submitted(route, base, work_dir, monkeypatch)
    )
    async with recorded as submission:
        assert json.loads(submission.record.read_text(encoding="utf-8"))["status"] in {
            "queued",
            "running",
        }
        caplog.set_level(logging.INFO, logger="ltspice_mcp.events")
        caplog.clear()

        replayed, status = await submission.replay()
        await submission.session.job_registry.drain_pending()

        assert replayed is True
        assert status == "interrupted"
        assert json.loads(submission.record.read_text(encoding="utf-8"))["status"] == (
            "interrupted"
        )
        events = [
            getattr(entry, "ltspice_event", {}).get("event")
            for entry in caplog.records
            if entry.name == "ltspice_mcp.events"
        ]
        assert "submitted" not in events


@pytest.mark.asyncio
async def test_a_replay_no_record_answers_still_names_a_job_its_session_finds(committed: Any):
    """With persistence off, no store read answers a replay from the journal,
    so the receipt is built from the journal's copy. The job id it gives is
    then found by the session that replayed it, as every receipt's is. (A
    resume refuses to run without persistence, so only the journal replay
    reaches this.)"""
    async with _journaled(committed) as submission:
        submission.session.job_registry.persist_enabled = False
        job_id = json.loads(submission.record.read_text(encoding="utf-8"))["job_id"]
        replayed, _status = await submission.replay()
        assert replayed is True
        found = await resolve_job_async(job_id, submission.session)
        assert found.job_id == job_id


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
        for module, line in _named_uses(frozenset({"add_experiment_job", "adopt"}))
        if module not in _REGISTRARS
    )
    assert not stray, f"jobs registered around the runner and registry: {stray}"
