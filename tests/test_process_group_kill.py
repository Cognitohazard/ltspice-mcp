"""Signalling a process group by its leader's pid (``proc_kill.kill_process_group``).

A group's id is its leader's pid. Once the leader is reaped, the system can give
that pid to another process, which can lead a group of its own with the same
id, so a signal sent to the id then reaches a process this server never
started. These tests start real processes, let a leader exit and be reaped,
and show which signals land, by a recorder process that logs every signal it
gets.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from ltspice_mcp.lib.proc_kill import (
    ProcessPresence,
    child_start_marker,
    kill_process_group,
    process_identity_presence,
)
from tests.conftest import wait_until, written

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Windows has no process groups")

#: Logs each SIGUSR1 and SIGUSR2 it gets, one name per line, after writing
#: ``ready`` once both are blocked. It takes them one at a time with
#: ``sigwait``, which returns a signal already pending before one sent later,
#: so the log is in the order they arrived (a Python handler can be interrupted
#: by the next one and log after it).
_RECORDER = """
import pathlib, signal, sys
log = pathlib.Path(sys.argv[1])
wanted = {signal.SIGUSR1, signal.SIGUSR2}
signal.pthread_sigmask(signal.SIG_BLOCK, wanted)
pathlib.Path(sys.argv[2]).write_text("ready", encoding="utf-8")
while True:
    number = signal.sigwait(wanted)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(signal.Signals(number).name + "\\n")
"""


#: Starts a recorder in the leader's own group (no new session), writes the
#: recorder's pid, and exits.
_LEADER = """
import pathlib, subprocess, sys
recorder = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]])
pathlib.Path(sys.argv[4]).write_text(str(recorder.pid), encoding="utf-8")
"""


def _signals(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").split() if log.exists() else []


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[bytes], Path]]:
    """A session leader that logs each signal it gets."""
    log, ready = tmp_path / "signals", tmp_path / "ready"
    process = subprocess.Popen(
        [sys.executable, "-c", _RECORDER, str(log), str(ready)], start_new_session=True
    )
    try:
        wait_until(written(ready, str.strip), what="the recorder to install its handlers")
        yield process, log
    finally:
        process.kill()
        process.wait()


def _delivered(process: subprocess.Popen[bytes], log: Path) -> list[str]:
    """Every signal ``process`` has been sent so far, read once the recorder
    logged a SIGUSR2 sent now: a signal pending before it is delivered no later."""
    sent = _signals(log).count("SIGUSR2") + 1
    os.kill(process.pid, signal.SIGUSR2)

    def logged() -> list[str] | None:
        seen = _signals(log)
        return seen if seen.count("SIGUSR2") == sent else None

    return wait_until(logged, what="the recorder to log SIGUSR2")


def test_a_pid_that_names_another_process_is_not_signalled(
    recorder: tuple[subprocess.Popen[bytes], Path],
):
    """The leader exited and was reaped; the process holding a pid now is
    another one. Nothing is signalled through the leader's identity, whatever
    the pid names now."""
    leader = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    started = child_start_marker(leader)
    assert started is not None
    leader.wait()
    other, log = recorder

    # The recorder stands where the leader's pid would be once reused.
    assert kill_process_group(other.pid, started, signal.SIGUSR1) is False
    # The leader's own pid is free, or names whatever took it.
    assert kill_process_group(leader.pid, started, signal.SIGUSR1) is False
    assert _delivered(other, log) == ["SIGUSR2"]

    assert kill_process_group(other.pid, child_start_marker(other), signal.SIGUSR1) is True
    assert _delivered(other, log) == ["SIGUSR2", "SIGUSR1", "SIGUSR2"]


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="the start marker of an exited, unreaped process is read from /proc",
)
def test_an_unreaped_leader_names_its_group_though_its_owner_liveness_is_absent(tmp_path: Path):
    """Until it is reaped, an exited leader holds its pid, so its group is still
    the one it led and what it left there is signalled."""
    log, ready, recorder_pid = tmp_path / "signals", tmp_path / "ready", tmp_path / "recorder"
    leader = subprocess.Popen(
        [sys.executable, "-c", _LEADER, _RECORDER, str(log), str(ready), str(recorder_pid)],
        start_new_session=True,
    )
    started = child_start_marker(leader)
    member = psutil.Process(wait_until(written(recorder_pid, int), what="the recorder to start"))
    try:
        wait_until(written(ready, str.strip), what="the recorder to install its handlers")
        wait_until(
            lambda: psutil.Process(leader.pid).status() == psutil.STATUS_ZOMBIE,
            what="the leader to exit",
        )
        assert child_start_marker(leader) == started
        # It can run no more work, which an owner's liveness asks about, but
        # its pid still names it alone, which a signal to its group asks.
        assert process_identity_presence(leader.pid, started) is ProcessPresence.ABSENT
        assert kill_process_group(leader.pid, started, signal.SIGUSR1) is True
        wait_until(lambda: "SIGUSR1" in _signals(log), what="the group member to get SIGUSR1")
    finally:
        member.kill()
        leader.wait()
    # Reaped: the marker can no longer be read for it.
    assert child_start_marker(leader) is None
