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
import subprocess
import sys
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
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment
from tests.test_recovery_records import recovery_job

#: A start marker no process running now has: the process that held a pid
#: before it was reused.
_EARLIER_PROCESS = "linux:00000000-0000-0000-0000-000000000000:1"


def _exited_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


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
        "exited_pid": (_exited_pid(), None, False),
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
        assert waited["timed_out"] is True
        assert cancel["error"]["code"] == "cancel_not_authorized"
    else:
        assert (status["status"], waited["status"], on_disk) == ("interrupted",) * 3
        assert waited["timed_out"] is False
        assert "error" not in cancel and cancel["items"] == []


@pytest.mark.asyncio
async def test_the_job_this_process_runs_is_its_own_whatever_its_record_says(
    state_no_sim: SessionState, work_dir: Path
):
    """Ownership is the registry's: a live job here is cancelled without a
    token even when its record names another pid, and is never re-read."""
    job = _experiment(work_dir, _circuit(work_dir), status="running")
    job.owner_pid = _exited_pid()
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

_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"
#: The module that owns process identity.
_OWNER = "lib/store.py"


def _names_an_owner_pid(node: ast.AST) -> bool:
    """``x.owner_pid``, ``owner_pid``, or ``x.owner.pid``."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and inner.attr == "owner_pid":
            return True
        if isinstance(inner, ast.Name) and inner.id == "owner_pid":
            return True
        if (
            isinstance(inner, ast.Attribute)
            and inner.attr == "pid"
            and isinstance(inner.value, ast.Attribute | ast.Name)
            and getattr(inner.value, "attr", getattr(inner.value, "id", None)) == "owner"
        ):
            return True
    return False


def _calls_getpid(node: ast.AST) -> bool:
    return any(
        isinstance(inner, ast.Call)
        and (
            (isinstance(inner.func, ast.Attribute) and inner.func.attr == "getpid")
            or (isinstance(inner.func, ast.Name) and inner.func.id == "getpid")
        )
        for inner in ast.walk(node)
    )


def test_no_module_but_the_owner_compares_a_job_s_owner_pid_with_its_own() -> None:
    """``owner_pid == os.getpid()`` and ``owner_pid in (0, os.getpid())`` read
    a reused pid as this process; the registry (``JobRegistry.owns``) and the
    store's identity checks (``owner_liveness``, ``is_this_process``) answer
    instead."""
    found = sorted(
        f"{path.relative_to(_SRC).as_posix()}:{node.lineno}"
        for path in _SRC.rglob("*.py")
        if path.relative_to(_SRC).as_posix() != _OWNER
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Compare)
        and any(_names_an_owner_pid(side) for side in (node.left, *node.comparators))
        and any(_calls_getpid(side) for side in (node.left, *node.comparators))
    )
    assert not found, f"a job's owner decided by pid outside {_OWNER}: {found}"
