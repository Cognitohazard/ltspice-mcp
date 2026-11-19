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
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import pytest

from ltspice_mcp.lib.proc_kill import (
    ProcessPresence,
    child_start_marker,
    kill_process_group,
    process_identity_presence,
    process_start_marker,
)
from tests.conftest import wait_until, written

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Windows has no process groups")

#: Logs each SIGUSR1 and SIGUSR2 it gets, one name per line, after writing
#: ``ready`` once it handles both. Python's C-level handler writes each
#: signal's number to the wakeup pipe as the signal is delivered, and the
#: recorder logs what one read of the pipe returns in a single write. A system
#: delivers every pending signal before the recorder's code runs again, so
#: signals pending together arrive in one read; Linux runs their handlers in
#: reverse (it stacks one frame per signal), so a read is logged lowest number
#: first, the order in which a process takes pending signals. SIGUSR1 is the
#: lower number on every platform, so a SIGUSR1 pending when a SIGUSR2 is sent
#: is logged no later than that SIGUSR2.
#:
#: It neither blocks the signals nor uses ``signal.sigwait``. On a GitHub macOS
#: runner, two signals reaching a recorder waiting in ``sigwait`` lost the first
#: in forty-nine pairs of fifty, ``sigwait`` once returned a number that names
#: no signal, and a recorder that polled its blocked, pending signals saw a
#: SIGUSR1 sent by ``killpg`` only after a SIGUSR2 sent after it.
#: After ``ready`` it writes how it found each signal's disposition and mask,
#: which it inherited, for a failure to show.
_RECORDER = """
import os, pathlib, signal, sys, threading
log = pathlib.Path(sys.argv[1])
wanted = {signal.SIGUSR1, signal.SIGUSR2}
inherited = signal.pthread_sigmask(signal.SIG_BLOCK, ())
found = " ".join(
    f"{s.name}={signal.getsignal(s)!r}{'(blocked)' if s in inherited else ''}"
    for s in sorted(wanted)
)
reader, writer = os.pipe()
os.set_blocking(writer, False)
signal.set_wakeup_fd(writer, warn_on_full_buffer=False)
for number in wanted:
    signal.signal(number, lambda *_: None)
signal.pthread_sigmask(signal.SIG_UNBLOCK, wanted)
pathlib.Path(sys.argv[2]).write_text(
    f"ready {found} threads={threading.active_count()}", encoding="utf-8"
)
while True:
    names = [signal.Signals(n).name for n in sorted(os.read(reader, 64)) if n in wanted]
    if names:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("".join(name + "\\n" for name in names))
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


def _marker_now(pid: int) -> str:
    try:
        return process_start_marker(pid)
    except (psutil.Error, OSError, ValueError, IndexError) as exc:
        return f"unreadable ({type(exc).__name__}: {exc})"


@dataclass
class Recorder:
    """A recorder process, its files, and every group signal the test sent
    through ``kill_process_group``, so a failed wait can say what happened."""

    process: subprocess.Popen[bytes]
    log: Path
    ready: Path
    stderr: Path
    calls: list[str] = field(default_factory=list)

    def kill_group(self, pid: int, marker: str | None, sig: int) -> bool:
        """``kill_process_group``, recording what it was given and decided, and
        the marker ``pid`` names when it is called."""
        now = _marker_now(pid)
        sent = kill_process_group(pid, marker, sig)
        self.calls.append(
            f"kill_process_group({pid}, {marker!r}, {signal.Signals(sig).name})"
            f" {'sent' if sent else 'declined'}; pid {pid} named {now!r}"
        )
        return sent

    def diagnosis(self) -> str:
        """What the recorder, and the signals sent to it, show now."""
        status = self.process.poll()
        try:
            proc = psutil.Process(self.process.pid)
            state = f"{proc.status()}, {proc.num_threads()} threads"
        except psutil.Error as exc:
            state = type(exc).__name__
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
        lines = [
            f"recorder pid {self.process.pid}: "
            + ("alive" if status is None else f"exited with status {status}")
            + f" ({state})",
            f"recorder reported: {_read(self.ready)!r}",
            f"recorder stderr: {_read(self.stderr)!r}",
            f"recorder log: {_signals(self.log)}",
            f"group signals: {self.calls or 'none'}",
            "this process: "
            + ", ".join(
                f"{s.name}={signal.getsignal(s)!r}{'(blocked)' if s in mask else ''}"
                for s in (signal.SIGUSR1, signal.SIGUSR2)
            ),
        ]
        return "\n".join(lines)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[Recorder]:
    """A session leader that logs each signal it gets."""
    log, ready, stderr = tmp_path / "signals", tmp_path / "ready", tmp_path / "stderr"
    with stderr.open("wb") as errors:
        process = subprocess.Popen(
            [sys.executable, "-c", _RECORDER, str(log), str(ready)],
            start_new_session=True,
            stderr=errors,
        )
    try:
        wait_until(written(ready, str.strip), what="the recorder to install its handlers")
        yield Recorder(process, log, ready, stderr)
    finally:
        process.kill()
        process.wait()


def _logged(recorder: Recorder, name: str, count: int) -> list[str]:
    """The recorder's log, read once it holds ``count`` of signal ``name``."""

    def logged() -> list[str] | None:
        seen = _signals(recorder.log)
        return seen if seen.count(name) == count else None

    try:
        return wait_until(logged, what=f"the recorder to log {name}")
    except pytest.fail.Exception as exc:
        pytest.fail(f"{exc}\n{recorder.diagnosis()}")


def _delivered(recorder: Recorder) -> list[str]:
    """Every signal the recorder has been sent so far, read once it logged a
    SIGUSR2 sent now: a SIGUSR1 pending before it is logged no later."""
    sent = _signals(recorder.log).count("SIGUSR2") + 1
    os.kill(recorder.process.pid, signal.SIGUSR2)
    return _logged(recorder, "SIGUSR2", sent)


def test_a_pid_that_names_another_process_is_not_signalled(recorder: Recorder):
    """The leader exited and was reaped; the process holding a pid now is
    another one. Nothing is signalled through the leader's identity, whatever
    the pid names now."""
    leader = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    started = child_start_marker(leader)
    assert started is not None
    leader.wait()
    other = recorder.process

    # The recorder stands where the leader's pid would be once reused.
    assert recorder.kill_group(other.pid, started, signal.SIGUSR1) is False
    # The leader's own pid is free, or names whatever took it.
    assert recorder.kill_group(leader.pid, started, signal.SIGUSR1) is False
    assert _delivered(recorder) == ["SIGUSR2"]

    assert recorder.kill_group(other.pid, child_start_marker(other), signal.SIGUSR1) is True
    assert _logged(recorder, "SIGUSR1", 1) == ["SIGUSR2", "SIGUSR1"]


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
