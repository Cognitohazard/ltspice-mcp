"""Whose job is it: the process a record names, and whether it is this one.

A record names its owning process by pid and, since pids are reused (quickly on
Windows), by that process's start marker. A dead server's record that names a
pid this process now holds is a dead server's record: reconciled when read,
not waited on, not cancellable as this process's own. Whether this process
runs a job is answered by its registry, never by comparing pids.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import authorize_resume
from ltspice_mcp.lib.proc_kill import process_start_marker
from ltspice_mcp.lib.recovery_records import ProcessIdentity, RecoveryError
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import handle_jobs
from tests._source_scan import PACKAGE, parsed
from tests.conftest import exited_pid
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment
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


@pytest.mark.parametrize(
    ("owner", "authorized_without_token"),
    [("this_process", True), ("this_pid_earlier_process", False), ("live_peer", False)],
)
def test_resume_authority_follows_the_owning_process_not_its_pid(
    owner: str, authorized_without_token: bool, work_dir: Path, live_peer_pid: int
):
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

    assert authorize_resume(job, job.control_token) == job.control_token
    if authorized_without_token:
        assert authorize_resume(job, None) == job.control_token
    else:
        with pytest.raises(RecoveryError) as refused:
            authorize_resume(job, None)
        assert refused.value.code == "resume_not_authorized"


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


def _is_getpid(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Attribute) and node.func.attr == "getpid")
        or (isinstance(node.func, ast.Name) and node.func.id == "getpid")
    )


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


def _pid_comparisons_in(tree: ast.AST) -> list[int]:
    """The lines of ``tree`` that compare a job's owner pid with this process's
    pid, each read directly or through a variable or attribute bound to it."""
    owners = _holders_of(tree, _is_owner_pid)
    ours = _holders_of(tree, _is_getpid)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            _holds(one, _is_owner_pid, owners) and _holds(other, _is_getpid, ours)
            for one in (node.left, *node.comparators)
            for other in (node.left, *node.comparators)
            if one is not other
        )
    )


def test_no_module_but_the_owner_compares_a_job_s_owner_pid_with_its_own() -> None:
    """``owner_pid == os.getpid()`` and ``owner_pid in (0, os.getpid())``, each
    side read directly or through a variable, read a reused pid as this
    process; the registry (``JobRegistry.owns``) and the store's identity
    checks (``owner_liveness``, ``is_this_process``) answer instead."""
    found = [
        f"{path.relative_to(PACKAGE).as_posix()}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).as_posix() != _OWNER
        for line in _pid_comparisons_in(parsed(path))
    ]
    assert not found, f"a job's owner decided by pid outside {_OWNER}: {found}"


def test_the_owner_pid_rule_catches_what_it_names() -> None:
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
    )
    assert _pid_comparisons_in(tree) == [2, 4, 6, 9, 13, 15, 20]
