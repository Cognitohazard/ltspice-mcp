"""A job that outlives the process that submitted it.

`run_experiments(wait=False, detach=True)` hands one experiment to a process
spawned for it. Every test below drives a real spawned owner running real
ngspice, because the whole feature is about what happens across a process
boundary: an in-process double would prove nothing about ownership, about a
record another process can read, or about an owner that is killed.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from pathlib import Path
from typing import cast

import psutil
import pytest

from ltspice_mcp.api import Api, ApiValidationError
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi

pytestmark = pytest.mark.skipif(
    shutil.which("ngspice") is None,
    reason="ngspice not on PATH",
)

# A divider that solves instantly: for the tests about the hand-off itself,
# where the run being over quickly is a convenience.
FAST_DECK = "* dc divider\nV1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.dc V1 0 10 5\n.end\n"

# A transient long enough that a case is still in flight while a test cancels
# or kills its owner: about half a second of real ngspice per case, run one at
# a time, so a four-case job takes a couple of seconds.
SLOW_DECK = (
    "* slow rc\nV1 in 0 SIN(0 1 1k)\nR1 in out 1k\nC1 out 0 1u\n.save V(out)\n.tran 20u 8\n.end\n"
)
SLOW_VARIATIONS = [{"kind": "assign", "assign": {"R1": ["1k", "2k", "3k", "4k"]}}]

#: Long enough to cover a cold interpreter start plus staging in the owner.
HANDOFF_TIMEOUT_S = 120.0

#: How long a record gets to stop claiming a dead owner is still running. No
#: process has to start for that, so a generous bound here would only make a
#: regression take two minutes to report.
RECLASSIFY_TIMEOUT_S = 30.0


def _api(work_dir: Path) -> Api:
    return Api(
        working_dir=work_dir,
        simulator="ngspice",
        allowed_paths=[work_dir],
        # One case at a time, so a multi-case job stays in flight long enough
        # for a test to reach it.
        max_parallel_sims=1,
    )


def _deck(work_dir: Path, name: str, content: str) -> str:
    (work_dir / name).write_text(content)
    return name


def _detached(receipt: dict) -> dict:
    """The receipt's detached-owner observation, which must be the only one."""
    codes = [item["code"] for item in receipt["observations"]]
    assert "process_owned_job" not in codes, (
        "the owner's own 'this process owns it' note reached the caller, "
        f"for whom it is false: {receipt['observations']}"
    )
    matches = [item for item in receipt["observations"] if item["code"] == "detached_owner"]
    assert len(matches) == 1, receipt["observations"]
    return matches[0]


def _gone(pid: int) -> bool:
    """Whether a process has stopped running, counting an uncollected one.

    A process this interpreter spawned stays in the process table until it is
    collected, so ``pid_exists`` alone would call a dead owner alive.
    """
    if not psutil.pid_exists(pid):
        return True
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _until(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(0.05)


def _submit_slow(api: Api, work_dir: Path, request_id: str) -> dict:
    return api.run_experiments(
        wait=False,
        detach=True,
        request_id=request_id,
        circuits=[{"path": _deck(work_dir, "slow.cir", SLOW_DECK), "id": "slow"}],
        variations=SLOW_VARIATIONS,
    )


# ---------------------------------------------------------------------------
# Argument policy — no engine, no process
# ---------------------------------------------------------------------------


def test_detaching_a_waited_call_is_refused_with_both_ways_forward(
    state_no_sim: SessionState,
) -> None:
    api = SyncApi(state_no_sim)
    with pytest.raises(ApiValidationError) as caught:
        api.run_experiments(detach=True, request_id="never", circuits=[{"path": "d.cir"}])
    message = str(caught.value)
    assert "wait=False" in message
    assert "api.wait(job_id)" in message


def test_detaching_a_raw_page_call_is_refused(state_no_sim: SessionState) -> None:
    api = SyncApi(state_no_sim)
    with pytest.raises(ApiValidationError) as caught:
        api.run_experiments(
            wait=False,
            detach=True,
            raw_page=True,
            request_id="never",
            circuits=[{"path": "d.cir"}],
        )
    assert "raw_page" in str(caught.value)


def test_detach_must_be_a_bool(state_no_sim: SessionState) -> None:
    api = SyncApi(state_no_sim)
    # Cast away the annotation: the point is what a caller who ignored it gets.
    detach = cast(bool, "yes")
    with pytest.raises(TypeError, match="detach must be a bool"):
        api.run_experiments(detach=detach, request_id="never", circuits=[{"path": "d.cir"}])


# ---------------------------------------------------------------------------
# The hand-off
# ---------------------------------------------------------------------------


def test_a_detached_job_outlives_the_session_that_submitted_it(work_dir: Path) -> None:
    api = _api(work_dir)
    receipt = api.run_experiments(
        wait=False,
        detach=True,
        request_id="detach-outlives",
        circuits=[{"path": _deck(work_dir, "fast.cir", FAST_DECK), "id": "div"}],
    )
    observation = _detached(receipt)
    owner_pid = observation["evidence"]["owner_pid"]
    log_file = Path(observation["evidence"]["log_file"])
    job_id = receipt["job_id"]

    assert owner_pid != os.getpid()
    assert str(owner_pid) in observation["detail"]
    assert str(log_file) in observation["detail"]
    assert log_file.is_file()

    # Closing the session that submitted it is what cancels a job this process
    # owns. This one is not ours, so it has to survive the close.
    api.close()

    with _api(work_dir) as fresh:
        final = fresh.wait(job_id, timeout=HANDOFF_TIMEOUT_S)
        assert not final.get("timed_out"), final
        assert final["status"] == "completed", final
        assert final["completeness"]["produced"] == 1, final["completeness"]

        # The result itself, not just the status: a job nobody supervised to
        # the end would leave a record with no raw behind it.
        analysis = fresh.analyze_results(
            sources=[{"job_id": job_id, "label": "div"}],
            recipes=[{"key": "vout", "metric": "value", "expr": "v(out)", "at": "10"}],
        )
        measured = analysis["results"]["vout"]["values"][0]["value"]
        assert measured["value"] == pytest.approx(5.0, rel=1e-6), measured

    _until(lambda: _gone(owner_pid), HANDOFF_TIMEOUT_S, "the detached owner to exit")


def test_replaying_a_detached_request_returns_the_same_job(work_dir: Path) -> None:
    request_id = "detach-replay"
    circuits = [{"path": _deck(work_dir, "fast.cir", FAST_DECK), "id": "div"}]
    with _api(work_dir) as api:
        first = api.run_experiments(
            wait=False, detach=True, request_id=request_id, circuits=circuits
        )
        job_id = first["job_id"]
        api.wait(job_id, timeout=HANDOFF_TIMEOUT_S)

        # In this process, the ordinary route: the request index already names
        # the job, so nothing is staged or submitted again.
        replayed = api.run_experiments(wait=False, request_id=request_id, circuits=circuits)
        assert replayed["job_id"] == job_id
        assert "idempotent_replay" in {item["code"] for item in replayed["observations"]}

        # And detached again: a second owner is spawned, takes the same replay
        # path, and hands back the same job rather than submitting a new one.
        again = _detached(
            api.run_experiments(wait=False, detach=True, request_id=request_id, circuits=circuits)
        )

    records = sorted(Store(work_dir).experiments_dir.glob("*.json"))
    assert [path.stem for path in records] == [job_id]
    # The owner named on the replay is the process that actually owns the
    # record, not the owner just spawned to look it up.
    assert again["evidence"]["owner_pid"] != os.getpid()


def test_cancelling_a_detached_job_from_another_session_stops_its_owner(
    work_dir: Path,
) -> None:
    api = _api(work_dir)
    receipt = _submit_slow(api, work_dir, "detach-cancel")
    job_id = receipt["job_id"]
    control_token = receipt["control_token"]
    owner_pid = _detached(receipt)["evidence"]["owner_pid"]
    api.close()

    with _api(work_dir) as fresh:
        # A different owner pid, so this is the foreign-owner path: without the
        # receipt's control token there is no authority to cancel at all.
        cancelled = fresh.jobs(action="cancel", job_id=job_id, control_token=control_token)
        assert cancelled["outcome"] != "failed", cancelled

        final = _until(
            lambda: fresh.jobs(action="status", job_id=job_id),
            HANDOFF_TIMEOUT_S,
            "the cancelled job to report",
        )
        assert final["status"] == "cancelled", final

    _until(lambda: _gone(owner_pid), HANDOFF_TIMEOUT_S, "the cancelled owner to exit")


def test_killing_a_detached_owner_leaves_an_interrupted_job(work_dir: Path) -> None:
    api = _api(work_dir)
    receipt = _submit_slow(api, work_dir, "detach-kill")
    job_id = receipt["job_id"]
    owner_pid = _detached(receipt)["evidence"]["owner_pid"]

    # The owner runs in its own session, so this takes its simulator with it —
    # the shape of a machine losing the whole process group, not a tidy exit.
    os.killpg(os.getpgid(owner_pid), signal.SIGKILL)

    # Read it back from the session that spawned the owner and has not
    # collected it. Its pid is still in the process table, and a job whose
    # owner is gone must not read as one that is still running.
    interrupted = _until(
        lambda: (
            status
            if (status := api.jobs(action="status", job_id=job_id))["status"] != "running"
            else None
        ),
        RECLASSIFY_TIMEOUT_S,
        "the killed owner's job to stop reporting as running",
    )
    assert interrupted["status"] == "interrupted", interrupted
    api.close()

    with _api(work_dir) as fresh:
        assert fresh.jobs(action="status", job_id=job_id)["status"] == "interrupted"


def test_the_owner_holds_its_own_engine_lease_and_releases_it(work_dir: Path) -> None:
    api = _api(work_dir)
    try:
        receipt = api.run_experiments(
            wait=False,
            detach=True,
            request_id="detach-lease",
            circuits=[{"path": _deck(work_dir, "fast.cir", FAST_DECK), "id": "div"}],
        )
        # The owner submitted while this process held the only lease this
        # process can hand out — the lease is per process, and the owner is
        # one of its own.
        with pytest.raises(Exception, match="already active in this process"):
            _api(work_dir)

        owner = api._detached_children[-1]
        assert owner.wait(timeout=HANDOFF_TIMEOUT_S) == 0
        log = Store(work_dir).detached_log("detach-lease").read_text(encoding="utf-8")
        assert "finished with status completed" in log
        assert "Traceback" not in log
    finally:
        api.close()

    # Nothing the owner did stopped this process from opening a session again.
    with _api(work_dir) as fresh:
        assert fresh.jobs(action="status", job_id=receipt["job_id"])["status"] == "completed"
