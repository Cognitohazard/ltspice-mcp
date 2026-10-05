"""Seeded schedule jitter: make a race lose on Linux the way it loses on Windows.

Every race the suite has hit was present on Linux too. It lost only on the
Windows runner, which hands work between threads more slowly, fires timers
up to 15.6 ms early, and starts processes slowly. With ``--jitter-seed=N``
(or ``LTSPICE_MCP_TEST_JITTER_SEED=N``) each test runs with three such
perturbations, drawn from a generator seeded by N and the test's id:

- **Thread to loop hand-offs** are delayed. ``loop.call_soon_threadsafe`` is
  the one door through which a worker thread reaches the event loop: an
  ``asyncio.to_thread`` result, a simulator's completion callback, a
  ``run_coroutine_threadsafe`` call, a child process's exit. Delaying it
  stretches the gap between a thread finishing and the loop acting on it,
  which is where a test that waits on a side effect reads too early. Calls
  from one thread keep their order, as asyncio guarantees.
- **Timers fire up to one Windows clock resolution early**, as asyncio does on
  Windows: a loop's ``_clock_resolution`` is set to 15.625 ms.
- **Process starts are delayed** before ``subprocess.Popen`` spawns.

The seed fixes the delays drawn, not the operating system's own scheduling, so
a failure under a seed usually reproduces under it but is not guaranteed to.
``--jitter-seed=random`` draws a seed and prints it in the report header.

The hooks are imported into ``tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
import os
import random
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

SEED_ENV = "LTSPICE_MCP_TEST_JITTER_SEED"
WINDOWS_CLOCK_RESOLUTION_S = 0.015625
"""asyncio runs a timer once it is within one clock resolution of now; on
Windows that resolution is the 15.6 ms system tick."""

HANDOFF_PROBABILITY = 0.5
HANDOFF_MAX_DELAY_S = 0.02
SPAWN_MAX_DELAY_S = 0.03

_SEED = pytest.StashKey[str | None]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--jitter-seed",
        default=os.environ.get(SEED_ENV) or None,
        help=(
            "Perturb thread hand-offs, timer precision and process starts with "
            "delays drawn from this seed ('random' draws one). Also read from "
            f"{SEED_ENV}."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    seed = config.getoption("--jitter-seed")
    if seed == "random":
        # Drawn once, by the controller: xdist starts its workers after this
        # runs, and they inherit the environment it sets.
        resolved = f"{SEED_ENV}_RESOLVED"
        seed = os.environ.get(resolved) or str(random.SystemRandom().randrange(1, 2**31))
        os.environ[resolved] = seed
    config.stash[_SEED] = None if seed is None else str(seed)


def _seed(config: pytest.Config) -> str | None:
    return config.stash.get(_SEED, None)


def pytest_report_header(config: pytest.Config) -> str | None:
    seed = _seed(config)
    return f"schedule jitter seed: {seed}" if seed is not None else None


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> None:
    seed = _seed(item.config)
    if seed is not None and call.excinfo is not None:
        item.add_report_section(
            call.when, "jitter", f"rerun with --jitter-seed={seed} to draw the same delays"
        )


class _Jitter:
    """The delays one test draws, from one generator shared by its threads."""

    def __init__(self, seed: str, test_id: str) -> None:
        self._rng = random.Random(f"{seed}:{test_id}")
        self._lock = threading.Lock()
        self._last_due: dict[tuple[int, int], float] = {}

    def draw(self, probability: float, max_delay: float) -> float:
        with self._lock:
            if self._rng.random() >= probability:
                return 0.0
            return self._rng.uniform(0.0, max_delay)

    def due(self, loop: asyncio.AbstractEventLoop, delay: float) -> float | None:
        """The loop time to run a hand-off at, or None to run it as asyncio would.

        Per (loop, calling thread), each hand-off is due strictly after the
        previous one, so the calls one thread makes keep their order.
        """
        key = (id(loop), threading.get_ident())
        with self._lock:
            now = loop.time()
            previous = self._last_due.get(key, 0.0)
            if delay <= 0.0 and previous < now:
                return None
            due = max(now + delay, previous + 1e-6)
            self._last_due[key] = due
            return due


@pytest.fixture(autouse=True)
def _schedule_jitter(request: pytest.FixtureRequest) -> Iterator[None]:
    seed = _seed(request.config)
    if seed is None:
        yield
        return
    jitter = _Jitter(seed, request.node.nodeid)
    patch = pytest.MonkeyPatch()

    base = asyncio.base_events.BaseEventLoop
    handoff = base.call_soon_threadsafe

    def delayed_handoff(
        self: asyncio.base_events.BaseEventLoop,
        callback: Any,
        *args: Any,
        context: Any = None,
    ) -> asyncio.Handle:
        if getattr(self, "_thread_id", None) == threading.get_ident():
            return handoff(self, callback, *args, context=context)
        due = jitter.due(self, jitter.draw(HANDOFF_PROBABILITY, HANDOFF_MAX_DELAY_S))
        if due is None:
            return handoff(self, callback, *args, context=context)
        return handoff(self, lambda: self.call_at(due, callback, *args, context=context))

    init = base.__init__

    def windows_clock(self: asyncio.base_events.BaseEventLoop, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        self._clock_resolution = WINDOWS_CLOCK_RESOLUTION_S  # pyright: ignore[reportAttributeAccessIssue]

    popen_init = subprocess.Popen.__init__

    def delayed_spawn(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
        delay = jitter.draw(1.0, SPAWN_MAX_DELAY_S)
        if delay:
            time.sleep(delay)
        popen_init(self, *args, **kwargs)

    patch.setattr(base, "call_soon_threadsafe", delayed_handoff)
    patch.setattr(base, "__init__", windows_clock)
    patch.setattr(subprocess.Popen, "__init__", delayed_spawn)
    try:
        yield
    finally:
        patch.undo()
