"""Tests for RunnerManager caching and invalidation logic."""

import asyncio
import contextlib
import types as _types
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.lib.runner_manager import _RUNNER_IMPORTS, RunnerManager


class _StubRunner:
    """Minimal stub to replace the real runner class."""

    def __init__(self, loop, simulator_class, output_folder, max_parallel):
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel
        self.busy = False
        self.owned_jobs: set[str] = set()

    def has_active_work(self) -> bool:
        return self.busy

    def owns_experiment_job(self, job_id: str) -> bool:
        return job_id in self.owned_jobs


@pytest.fixture(autouse=True)
def _patch_runner_imports(monkeypatch, request):
    """Patch importlib.import_module so RunnerManager creates _StubRunner instances.

    A test class that sets ``real_runners`` gets the real runner classes.
    """
    if getattr(request.cls, "real_runners", False):
        return
    stub_module = MagicMock()
    stub_module.ExperimentRunner = _StubRunner

    import importlib

    original = importlib.import_module
    runner_modules = {value[0] for value in _RUNNER_IMPORTS.values()}

    def patched(name):
        if name in runner_modules:
            return stub_module
        return original(name)

    monkeypatch.setattr("importlib.import_module", patched)


@pytest.fixture
def loop():
    """Provide a fresh event loop, closed after each test."""
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


class TestRunnerManager:
    def test_same_context_returns_cached(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        out = Path("/tmp/out")

        r1 = mgr.get_experiment_runner(loop, sim_cls, out)
        r2 = mgr.get_experiment_runner(loop, sim_cls, out)
        assert r1 is r2

    def test_loop_change_invalidates(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        out = Path("/tmp/out")

        r1 = mgr.get_experiment_runner(loop, sim_cls, out)

        loop2 = asyncio.new_event_loop()
        try:
            r2 = mgr.get_experiment_runner(loop2, sim_cls, out)
            assert r1 is not r2
        finally:
            loop2.close()

    def test_simulator_change_invalidates(self, loop):
        mgr = RunnerManager()
        out = Path("/tmp/out")

        cls_a = type("SimA", (), {})
        cls_b = type("SimB", (), {})

        r1 = mgr.get_experiment_runner(loop, cls_a, out)
        r2 = mgr.get_experiment_runner(loop, cls_b, out)
        assert r1 is not r2

    def test_output_folder_change_invalidates(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})

        r1 = mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/a"))
        r2 = mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/b"))
        assert r1 is not r2

    def test_reset_clears_everything(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/out"))
        assert len(mgr._runners) > 0

        mgr.reset()
        assert len(mgr._runners) == 0
        assert mgr._loop is None


def _submitter():
    """A fresh manager's lookup for one simulator and folder, by cap."""
    mgr = RunnerManager()
    loop = asyncio.get_running_loop()
    sim_cls = type("FakeSim", (), {})
    return lambda max_parallel: mgr.get_experiment_runner(
        loop, sim_cls, Path("runs"), max_parallel=max_parallel
    )


async def _admitted_without_waiting(runner) -> bool:
    """Whether one more launch permit is granted at once. A granted permit is
    left held; a wait is cancelled."""
    attempt = asyncio.ensure_future(runner.acquire_launch_slot())
    await asyncio.sleep(0)
    if attempt.done():
        return True
    attempt.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await attempt
    return False


class TestCapChangeOnTheRealRunner:
    """A later submission's cap, applied in place by the real runner's setter.

    The cached instance is kept so an in-flight job's cancel state survives;
    the cap was once honored only at creation, so a second run silently used
    the first's (observed: 4 processes under a requested 2).
    """

    real_runners = True

    async def test_a_lowered_cap_waits_for_held_permits_then_holds(self):
        submit = _submitter()
        runner = submit(2)
        await runner.acquire_launch_slot()
        await runner.acquire_launch_slot()

        assert submit(1) is runner
        assert runner.max_parallel == 1
        # The two admitted runs keep their permits, and nothing joins them.
        assert not await _admitted_without_waiting(runner)
        runner.release_launch_slot()
        runner.release_launch_slot()

        # The next submission finds the runner idle, and the cap now holds.
        assert submit(1) is runner
        assert await _admitted_without_waiting(runner)
        assert not await _admitted_without_waiting(runner)
        runner.release_launch_slot()
        assert await _admitted_without_waiting(runner)
        runner.release_launch_slot()

    async def test_a_lowered_cap_applies_once_the_runner_drains(self):
        submit = _submitter()
        runner = submit(2)
        await runner.acquire_launch_slot()
        await runner.acquire_launch_slot()
        submit(1)
        runner.release_launch_slot()
        runner.release_launch_slot()

        # No further submission: draining alone puts the lowered cap in force.
        assert await _admitted_without_waiting(runner)
        assert not await _admitted_without_waiting(runner), (
            "a second run started under a cap of one"
        )
        runner.release_launch_slot()

    async def test_a_lowered_cap_applies_when_the_last_waiter_gives_up(self):
        """The last claim on a permit can end in a cancelled wait instead of a
        release: a waiter handed the final permit, cancelled before it resumes.
        The runner is idle then too, so the lowered cap applies."""
        submit = _submitter()
        runner = submit(2)
        await runner.acquire_launch_slot()
        await runner.acquire_launch_slot()
        waiter = asyncio.ensure_future(runner.acquire_launch_slot())
        await asyncio.sleep(0)
        submit(1)
        runner.release_launch_slot()  # hands this permit to the waiter
        runner.release_launch_slot()
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter

        assert await _admitted_without_waiting(runner)
        assert not await _admitted_without_waiting(runner), (
            "a second run started under a cap of one"
        )
        runner.release_launch_slot()

    async def test_a_lookup_while_a_permit_is_handed_over_keeps_the_cap(self):
        """Releasing the only permit wakes the next run in line, which has not
        counted itself yet. A submission that looks the runner up in that gap
        must not reset the permits, or the woken run and a new one would both
        launch under a cap of one."""
        submit = _submitter()
        runner = submit(1)
        await runner.acquire_launch_slot()
        woken = asyncio.ensure_future(runner.acquire_launch_slot())
        await asyncio.sleep(0)
        assert not woken.done()

        runner.release_launch_slot()  # hands the permit to the waiting run
        assert submit(1) is runner
        await woken
        assert not await _admitted_without_waiting(runner), (
            "a second run started under a cap of one"
        )
        runner.release_launch_slot()


class TestCapEviction:
    """The LRU cap must never evict a runner with in-flight work — dropping it
    would split its concurrency semaphore and lose per-job cancel state."""

    def _fill_to_cap(self, mgr, loop, sim_cls) -> list[_StubRunner]:
        from ltspice_mcp.lib.runner_manager import _RUNNER_CACHE_CAP

        # The patched import hands out stubs, whatever the manager's type says.
        return [
            cast(_StubRunner, mgr.get_experiment_runner(loop, sim_cls, Path(f"/tmp/out{i}")))
            for i in range(_RUNNER_CACHE_CAP)
        ]

    def test_oldest_idle_runner_evicted_busy_survive(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        runners = self._fill_to_cap(mgr, loop, sim_cls)
        runners[0].busy = True
        runners[1].busy = True

        mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/overflow"))

        cached = set(mgr._runners.values())
        assert runners[0] in cached and runners[1] in cached
        assert runners[2] not in cached  # oldest idle went, not the busy heads

    def test_no_eviction_when_all_busy(self, loop):
        from ltspice_mcp.lib.runner_manager import _RUNNER_CACHE_CAP

        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        runners = self._fill_to_cap(mgr, loop, sim_cls)
        for runner in runners:
            runner.busy = True

        mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/overflow"))

        cached = set(mgr._runners.values())
        assert all(runner in cached for runner in runners)  # cache exceeds cap instead
        assert len(mgr._runners) == _RUNNER_CACHE_CAP + 1


class TestExperimentRunnerRouting:
    """A job's cancel state lives on the coordinator that launched it; with
    several runners cached, most-recent is not necessarily the owner."""

    def test_owner_is_found_among_several(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        older = cast(_StubRunner, mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/a")))
        newer = mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/b"))
        older.owned_jobs.add("exp_1")

        job = _types.SimpleNamespace(job_id="exp_1")
        assert mgr.get_experiment_runner_for(job) is older
        assert mgr.get_experiment_runner_for(job) is not newer

    def test_none_when_no_runner_owns_it(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        mgr.get_experiment_runner(loop, sim_cls, Path("/tmp/a"))

        job = _types.SimpleNamespace(job_id="exp_unknown")
        assert mgr.get_experiment_runner_for(job) is None
