"""Whose job is it: the process a record names, and whether it is this one.

A record names its owning process by pid and, since pids are reused (quickly on
Windows), by that process's start marker. A dead server's record that names a
pid this process now holds is a dead server's record: reconciled when read,
not waited on, not cancellable as this process's own. Whether this process
runs a job is answered by its registry, for a cancel and a resume alike, never
by comparing the identity a record names with this process's.
"""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import authorize_resume
from ltspice_mcp.lib.experiment_runner import AdmissionResult, ExperimentRunner
from ltspice_mcp.lib.proc_kill import process_start_marker
from ltspice_mcp.lib.recovery_records import ProcessIdentity, RecoveryError
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import handle_jobs
from tests._source_scan import PACKAGE, parsed
from tests.conftest import LIVENESS_S, coordinator_returned, exited_pid
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment
from tests.test_recovery_launch import RecordedNGspice, _process
from tests.test_recovery_launch import committed as committed
from tests.test_recovery_records import recovery_job

#: A start marker no process running now has: the process that held a pid
#: before it was reused.
_EARLIER_PROCESS = "linux:00000000-0000-0000-0000-000000000000:1"


def _owners(live_peer_pid: int) -> dict[str, tuple[int, str | None, bool]]:
    """Each kind of record owner: its pid, the start marker the record holds
    (None for a record written before markers were), and whether that process
    is still running."""
    import os

    this = os.getpid()
    return {
        "this_pid_earlier_process": (this, _EARLIER_PROCESS, False),
        "this_pid_unmarked": (this, None, False),
        "this_process_another_session": (this, process_start_marker(this), True),
        "exited_pid": (exited_pid(), None, False),
        "live_peer": (live_peer_pid, None, True),
        "live_peer_marked": (live_peer_pid, process_start_marker(live_peer_pid), True),
        "live_peer_pid_reused": (live_peer_pid, _EARLIER_PROCESS, False),
    }


def _write_running_record(work_dir: Path, job_id: str, pid: int, marker: str | None) -> Path:
    """A running job's record, as the process ``pid`` would have left it."""
    job = _experiment(work_dir, _circuit(work_dir), job_id=job_id, status="running")
    job.owner_pid = pid
    experiment_store.save_job(job)
    record = json.loads(job.store_path.read_text(encoding="utf-8"))
    if marker is None:
        record.pop("owner_start", None)
    else:
        record["owner_start"] = {"pid": pid, "start_marker": marker}
    job.store_path.write_text(json.dumps(record), encoding="utf-8")
    return job.store_path


_OWNER_KINDS = (
    "this_pid_earlier_process",
    "this_pid_unmarked",
    "this_process_another_session",
    "exited_pid",
    "live_peer",
    "live_peer_marked",
    "live_peer_pid_reused",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _OWNER_KINDS)
async def test_a_record_s_owner_answers_load_wait_and_cancel_alike(
    kind: str, state_no_sim: SessionState, work_dir: Path, live_peer_pid: int
):
    """A session that runs nothing reads each record. A live owner's job stays
    running: a wait re-reads its record, and a cancel needs its token. A gone
    owner's job is reconciled as it is read, written back, and finished."""
    pid, marker, alive = _owners(live_peer_pid)[kind]
    job_id = f"exp_owner_{kind}"
    record = _write_running_record(work_dir, job_id, pid, marker)

    status = _assert_jobs_schema(await handle_jobs(_args("status", job_id=job_id), state_no_sim))
    waited = _assert_jobs_schema(
        await handle_jobs(_args("wait", job_id=job_id, timeout_s=0), state_no_sim)
    )
    cancel = _assert_jobs_schema(await handle_jobs(_args("cancel", job_id=job_id), state_no_sim))
    await state_no_sim.job_registry.drain_pending()
    on_disk = json.loads(record.read_text(encoding="utf-8"))["status"]

    if alive:
        assert (status["status"], waited["status"], on_disk) == ("running",) * 3
        assert waited.get("timed_out", False) is True
        assert cancel["error"]["code"] == "cancel_not_authorized"
    else:
        assert (status["status"], waited["status"], on_disk) == ("interrupted",) * 3
        assert waited.get("timed_out", False) is False
        assert "error" not in cancel and cancel.get("items", []) == []


@pytest.mark.asyncio
async def test_the_job_this_process_runs_is_its_own_whatever_its_record_says(
    state_no_sim: SessionState, work_dir: Path
):
    """Ownership is the registry's: a live job here is cancelled without a
    token even when its record names another pid, and is never re-read."""
    job = _experiment(work_dir, _circuit(work_dir), status="running")
    job.owner_pid = exited_pid()
    state_no_sim.job_registry.reserve(job)
    state_no_sim.add_experiment_job(job, already_persisted=True)

    assert state_no_sim.job_registry.owns(job)
    assert state_no_sim.job_registry.refresh_foreign_job(job) is job
    cancel = _assert_jobs_schema(
        await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim)
    )
    # No coordinator runs it here, which is what the refusal says; it is not
    # refused for want of a token.
    assert cancel["error"]["code"] == "cancel_unavailable"


@pytest.mark.parametrize("owner", ["this_process", "this_pid_earlier_process", "live_peer"])
def test_no_identity_a_record_names_authorizes_a_resume(
    owner: str, work_dir: Path, live_peer_pid: int
):
    """Not even this process's own: the session that runs the job resumes it
    without the token, and every other caller needs it, whatever process the
    record names."""
    import os

    identity = {
        "this_process": ProcessIdentity(os.getpid(), process_start_marker(os.getpid())),
        "this_pid_earlier_process": ProcessIdentity(os.getpid(), _EARLIER_PROCESS),
        "live_peer": ProcessIdentity(live_peer_pid, process_start_marker(live_peer_pid)),
    }[owner]
    job = recovery_job(work_dir)
    assert job.recovery is not None
    job.recovery = replace(job.recovery, owner=identity)
    job.owner_pid = identity.pid

    for owned in (True, False):
        assert authorize_resume(job, job.control_token, owned=owned) == job.control_token
    assert authorize_resume(job, None, owned=True) == job.control_token
    with pytest.raises(RecoveryError) as refused:
        authorize_resume(job, None, owned=False)
    assert refused.value.code == "resume_not_authorized"


#: The refusals a caller gets for want of authority over a job.
_NOT_AUTHORIZED = frozenset({"cancel_not_authorized", "resume_not_authorized"})


async def test_cancel_and_resume_ask_the_session_that_runs_the_job(
    committed, monkeypatch: pytest.MonkeyPatch
):
    """Whether a caller may act on a job without its token is one answer: does
    the caller's session run the job (``JobRegistry.owns``)? A second session
    in the same process does not, though the job's record names this process
    by pid and start marker, so it needs the token to cancel and to resume
    alike; the session that runs the job needs it for neither."""
    _fixture_runner, request, job, store = committed
    owner = request.state
    other = SessionState.create(owner.config, available={})
    runner = owner.runners.get_experiment_runner(
        asyncio.get_running_loop(), RecordedNGspice, store.runs_root(), 1
    )
    launched, release = threading.Event(), threading.Event()

    def held_launch(_argv, _kwargs) -> None:
        launched.set()
        assert release.wait(LIVENESS_S)

    async def kill_releases(_runner, _token: str) -> None:
        release.set()

    _process(monkeypatch, inspect=held_launch)
    monkeypatch.setattr(ExperimentRunner, "_kill_case", kill_releases)
    ready = asyncio.get_running_loop().create_future()
    await runner.start_committed(request, AdmissionResult(job, False, True), ready)
    try:
        assert await asyncio.to_thread(launched.wait, LIVENESS_S)

        async def authorized(session: SessionState, action: str) -> bool:
            extra = {"resume_request_id": f"resume-{id(session)}"} if action == "resume" else {}
            data = _assert_jobs_schema(
                await handle_jobs(_args(action, job_id=job.job_id, **extra), session)
            )
            return data.get("error", {}).get("code") not in _NOT_AUTHORIZED

        # Resume first: the owner's cancel stops the job.
        answers = {
            (name, action): await authorized(session, action)
            for name, session in (("other", other), ("owner", owner))
            for action in ("resume", "cancel")
        }
        assert answers == {
            ("other", "resume"): False,
            ("other", "cancel"): False,
            ("owner", "resume"): True,
            ("owner", "cancel"): True,
        }
    finally:
        release.set()
        await coordinator_returned(owner, job)
        await owner.job_registry.drain_pending()


def test_a_record_this_process_writes_names_it_by_pid_and_start_marker(work_dir: Path):
    import os

    job = _experiment(work_dir, _circuit(work_dir), status="running")
    experiment_store.save_job(job)
    record = json.loads(Store(work_dir).job_record(job.job_id).read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()
    assert record["owner_start"] == {
        "pid": os.getpid(),
        "start_marker": process_start_marker(os.getpid()),
    }


# ---------------------------------------------------------------------------
# Ownership of the question
# ---------------------------------------------------------------------------

#: The module that owns process identity.
_OWNER = "lib/store.py"


def _is_owner_pid(node: ast.AST) -> bool:
    """``x.owner_pid``, ``owner_pid``, or ``x.owner.pid``."""
    if isinstance(node, ast.Attribute) and node.attr == "owner_pid":
        return True
    if isinstance(node, ast.Name) and node.id == "owner_pid":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "pid"
        and isinstance(node.value, ast.Attribute | ast.Name)
        and getattr(node.value, "attr", getattr(node.value, "id", None)) == "owner"
    )


def _names_owner(node: ast.AST) -> bool:
    """``owner`` or ``x.owner``."""
    return (isinstance(node, ast.Name) and node.id == "owner") or (
        isinstance(node, ast.Attribute) and node.attr == "owner"
    )


def _is_owner_marker(node: ast.AST) -> bool:
    """``x.owner_start``, ``owner_start``, or ``x.owner.start_marker``."""
    if isinstance(node, ast.Attribute) and node.attr == "owner_start":
        return True
    if isinstance(node, ast.Name) and node.id == "owner_start":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "start_marker"
        and _names_owner(node.value)
    )


def _is_owner_identity(node: ast.AST) -> bool:
    """The pid or the start marker a record names its owning process by."""
    return _is_owner_pid(node) or _is_owner_marker(node)


def _calls(node: ast.AST, *names: str) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Attribute) and node.func.attr in names)
        or (isinstance(node.func, ast.Name) and node.func.id in names)
    )


def _is_getpid(node: ast.AST) -> bool:
    return _calls(node, "getpid")


def _is_this_identity(node: ast.AST) -> bool:
    """This process's pid or start marker, read where it is read."""
    return _calls(node, "getpid", "this_process_start_marker", "process_start_marker")


def _holder(node: ast.AST) -> str | None:
    """The name a variable or an attribute (``self._pid``) is read or bound by."""
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _holds(node: ast.AST, kind: Callable[[ast.AST], bool], holders: set[str]) -> bool:
    """Whether ``node`` reads a value of ``kind`` itself or through a holder."""
    return any(kind(inner) or _holder(inner) in holders for inner in ast.walk(node))


def _binding(node: ast.AST) -> tuple[list[ast.expr], ast.expr] | None:
    """The targets and value of an ``=``, ``:=`` or annotated assignment."""
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
        return [node.target], node.value
    return None


def _holders_of(tree: ast.AST, kind: Callable[[ast.AST], bool]) -> set[str]:
    """The variables and attributes ``tree`` binds (``=``, ``:=``, an annotated
    assignment) to a value of ``kind``, or to another such variable."""
    bindings = [found for node in ast.walk(tree) if (found := _binding(node)) is not None]
    holders: set[str] = set()
    while True:
        grown = holders | {
            name
            for targets, value in bindings
            if _holds(value, kind, holders)
            for target in targets
            if (name := _holder(target)) is not None
        }
        if grown == holders:
            return holders
        holders = grown


def _identity_comparisons_in(tree: ast.AST) -> list[int]:
    """The lines of ``tree`` that compare the identity a job's record names its
    owner by (pid or start marker) with this process's, each side read
    directly or through a variable or attribute bound to it."""
    owners = _holders_of(tree, _is_owner_identity)
    ours = _holders_of(tree, _is_this_identity)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            _holds(one, _is_owner_identity, owners) and _holds(other, _is_this_identity, ours)
            for one in (node.left, *node.comparators)
            for other in (node.left, *node.comparators)
            if one is not other
        )
    )


def test_no_module_but_the_owner_compares_a_job_s_owner_identity_with_its_own() -> None:
    """``owner_pid == os.getpid()``, ``owner_pid in (0, os.getpid())`` and
    ``owner.start_marker == this_process_start_marker()``, each side read
    directly or through a variable, answer "does this process run the job?"
    from the record: a reused pid reads as this process, and so does a job
    another session of this process runs. The registry (``JobRegistry.owns``)
    answers that; a record's identity answers only whether the process it
    names is alive (``store.owner_liveness``)."""
    found = [
        f"{path.relative_to(PACKAGE).as_posix()}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).as_posix() != _OWNER
        for line in _identity_comparisons_in(parsed(path))
    ]
    assert not found, f"a job's owner decided by its identity outside {_OWNER}: {found}"


def test_the_owner_identity_rule_catches_what_it_names() -> None:
    tree = ast.parse(
        "def direct(job):\n"
        "    return job.owner_pid == os.getpid()\n"  # 2
        "def among(owner_pid):\n"
        "    return owner_pid in (0, os.getpid())\n"  # 4
        "def nested(job):\n"
        "    return job.owner.pid != getpid()\n"  # 6
        "def ours_in_a_variable(job):\n"
        "    me = os.getpid()\n"
        "    return job.owner_pid == me\n"  # 9
        "def theirs_in_a_variable(job):\n"
        "    owner = job.owner_pid\n"
        "    copied = owner\n"
        "    return copied == os.getpid()\n"  # 13
        "def walrus(record):\n"
        "    return (mine := os.getpid()) == record.owner_pid\n"  # 15
        "class Session:\n"
        "    def __init__(self):\n"
        "        self._pid: int = os.getpid()\n"
        "    async def owns(self, job):\n"
        "        return job.owner_pid == self._pid\n"  # 20
        "def unrelated(other_pid, job, recorded):\n"
        "    pid = os.getpid()\n"
        "    return pid == other_pid, job.owner_pid == recorded\n"
        "def marked(job):\n"
        "    owner = job.recovery.owner\n"
        "    return owner.start_marker == this_process_start_marker()\n"  # 26
        "def marked_on_the_record(record):\n"
        "    mine = process_start_marker(os.getpid())\n"
        "    return record.owner_start != (os.getpid(), mine)\n"  # 29
        "def two_records(fresh, stale):\n"
        "    marker = this_process_start_marker()\n"
        "    return fresh.owner != stale.owner, marker is None\n"
    )
    assert _identity_comparisons_in(tree) == [2, 4, 6, 9, 13, 15, 20, 26, 29]
