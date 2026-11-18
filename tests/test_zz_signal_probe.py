"""Temporary probe: how often a recorder misses a signal on this platform."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import psutil
import pytest

from tests.conftest import wait_until, written
from tests.test_process_group_kill import _RECORDER, _signals

pytestmark = pytest.mark.skipif(os.name != "posix", reason="probe")

_PROBE_WAIT = 4.0
_ROUNDS = 150


def proc_err(proc: subprocess.Popen[bytes]) -> str:
    return (
        ""
        if proc.stderr is None
        else proc.stderr.read().decode(errors="replace")[-300:]
        if proc.poll() is not None
        else "alive"
    )


def _one(tmp: Path, variant: str) -> str | None:
    log, ready = tmp / "signals", tmp / "ready"
    proc = subprocess.Popen(
        [sys.executable, "-c", _RECORDER, str(log), str(ready)],
        start_new_session=True,
        stderr=subprocess.PIPE,
    )
    try:
        report = wait_until(written(ready, str.strip), what="ready")
        threads = psutil.Process(proc.pid).num_threads()
        if variant == "kill+kill":
            os.kill(proc.pid, signal.SIGUSR1)
            os.kill(proc.pid, signal.SIGUSR2)
        elif variant == "killpg+kill":
            os.killpg(proc.pid, signal.SIGUSR1)
            os.kill(proc.pid, signal.SIGUSR2)
        elif variant == "killpg,wait,kill":
            os.killpg(proc.pid, signal.SIGUSR1)
            wait_until(lambda: "SIGUSR1" in _signals(log), what="usr1")
            os.kill(proc.pid, signal.SIGUSR2)
        try:
            wait_until(lambda: len(_signals(log)) >= 2, timeout_s=_PROBE_WAIT, what="both")
            proc.kill()
            proc.wait()
            err = proc_err(proc)
            return f"spurious {variant}: {err!r}" if err else None
        except pytest.fail.Exception:
            return (
                f"{variant}: log={_signals(log)} alive={proc.poll() is None} "
                f"os_threads={threads} stderr={proc_err(proc)!r} "
                f"report={report!r}"
            )
    finally:
        proc.kill()
        proc.wait()


def test_probe_signal_loss(tmp_path: Path):
    misses: list[str] = []
    counts: Counter[str] = Counter()
    for i in range(_ROUNDS):
        for variant in ("kill+kill", "killpg+kill", "killpg,wait,kill"):
            sub = tmp_path / f"{i}-{variant.replace(',', '_').replace('+', '_')}"
            sub.mkdir()
            miss = _one(sub, variant)
            counts[variant] += 1
            if miss:
                misses.append(miss)
    summary = Counter(m.split(":")[0] for m in misses)
    pytest.fail(
        f"PROBE {sys.platform}: rounds={dict(counts)} misses={dict(summary)}\n"
        + "\n".join(misses[:12])
    )
