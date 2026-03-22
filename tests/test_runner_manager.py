"""Tests for RunnerManager caching and invalidation logic."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.lib.runner_manager import _RUNNER_IMPORTS, RunnerManager


class _StubRunner:
    """Minimal stub to replace real runner classes."""

    def __init__(self, loop, simulator_class, output_folder, max_parallel):
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder


@pytest.fixture(autouse=True)
def _patch_runner_imports(monkeypatch):
    """Patch importlib.import_module so RunnerManager creates _StubRunner instances."""
    stub_module = MagicMock()
    stub_module.SimulationRunner = _StubRunner
    stub_module.SweepRunner = _StubRunner
    stub_module.MonteCarloRunner = _StubRunner

    import importlib

    original = importlib.import_module
    runner_modules = {v[0] for v in _RUNNER_IMPORTS.values()}

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

        r1 = mgr.get_sim_runner(loop, sim_cls, out)
        r2 = mgr.get_sim_runner(loop, sim_cls, out)
        assert r1 is r2

    def test_loop_change_invalidates(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        out = Path("/tmp/out")

        r1 = mgr.get_sim_runner(loop, sim_cls, out)

        loop2 = asyncio.new_event_loop()
        try:
            r2 = mgr.get_sim_runner(loop2, sim_cls, out)
            assert r1 is not r2
        finally:
            loop2.close()

    def test_simulator_change_invalidates(self, loop):
        mgr = RunnerManager()
        out = Path("/tmp/out")

        cls_a = type("SimA", (), {})
        cls_b = type("SimB", (), {})

        r1 = mgr.get_sim_runner(loop, cls_a, out)
        r2 = mgr.get_sim_runner(loop, cls_b, out)
        assert r1 is not r2

    def test_output_folder_change_invalidates(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})

        r1 = mgr.get_sim_runner(loop, sim_cls, Path("/tmp/a"))
        r2 = mgr.get_sim_runner(loop, sim_cls, Path("/tmp/b"))
        assert r1 is not r2

    def test_reset_clears_everything(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        mgr.get_sim_runner(loop, sim_cls, Path("/tmp/out"))
        assert len(mgr._runners) > 0

        mgr.reset()
        assert len(mgr._runners) == 0
        assert mgr._loop is None

    def test_different_runner_types_coexist(self, loop):
        mgr = RunnerManager()
        sim_cls = type("FakeSim", (), {})
        out = Path("/tmp/out")

        sim = mgr.get_sim_runner(loop, sim_cls, out)
        sweep = mgr.get_sweep_runner(loop, sim_cls, out)
        assert sim is not sweep
        assert len(mgr._runners) == 2
