"""Centralized runner lifecycle management.

Owns SimulationRunner, SweepRunner, and MonteCarloRunner instances.
Replaces the fragile module-level singleton pattern where each tool module
independently checked for staleness (event loop, simulator, output folder).

All three runners share the same constructor signature and staleness
conditions. This class provides a single invalidation mechanism.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ltspice_mcp.lib.runner_base import DEFAULT_MAX_PARALLEL

if TYPE_CHECKING:
    from ltspice_mcp.lib.montecarlo_runner import MonteCarloRunner
    from ltspice_mcp.lib.sim_runner import SimulationRunner
    from ltspice_mcp.lib.sweep_runner import SweepRunner

logger = logging.getLogger(__name__)

# Import paths for lazy loading (avoids circular imports with state.py)
_RUNNER_IMPORTS: dict[str, tuple[str, str]] = {
    "sim": ("ltspice_mcp.lib.sim_runner", "SimulationRunner"),
    "sweep": ("ltspice_mcp.lib.sweep_runner", "SweepRunner"),
    "mc": ("ltspice_mcp.lib.montecarlo_runner", "MonteCarloRunner"),
}


_RUNNER_CACHE_CAP = 8
"""Upper bound on cached runner instances. Keys are (kind, simulator class,
output folder); distinct folders arise from relative-include decks running in
their own dirs, so the cache is LRU-bounded rather than unbounded."""


class RunnerManager:
    """Creates and caches runner instances, one per (kind, simulator, folder).

    Runners are cached per simulator class and output folder so a per-run
    simulator override (or a deck that runs in its own directory) does not
    evict a runner with in-flight work — eviction would drop its concurrency
    semaphore and per-job cancel events. Only an event-loop change (test
    fixtures) invalidates everything: runners bridge worker callbacks onto
    the loop they were created with.
    """

    def __init__(self) -> None:
        self._runners: dict[tuple[str, type, Path], Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_or_create(
        self,
        kind: str,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int,
    ) -> Any:
        """Get a cached runner or create a new one."""
        if self._loop is None or self._loop is not loop:
            self._runners.clear()
            self._loop = loop

        key = (kind, simulator_class, output_folder)
        runner = self._runners.pop(key, None)
        if runner is not None:
            # Re-insert to refresh LRU recency. A cached runner keeps the
            # max_parallel it was created with, but a later call may request a
            # different cap. Update it in place: each batch rebuilds its
            # spicelib SimRunner from ``self.max_parallel`` at launch, so the
            # new cap takes effect on the next batch this runner starts.
            # Updating the attribute (vs. recreating the instance) preserves
            # the per-job cancel-event / live-process map that an in-flight
            # batch — and cancel_job — depend on.
            runner.max_parallel = max_parallel
            self._runners[key] = runner
            return runner

        if len(self._runners) >= _RUNNER_CACHE_CAP:
            # Never evict a runner with in-flight work: dropping it would
            # split the concurrency semaphore (a recreated instance admits
            # max_parallel more jobs) and lose its per-job cancel state. If
            # every cached runner is busy, let the cache exceed the cap —
            # busy runners are bounded by running jobs, not by this dict.
            victim = next((k for k, r in self._runners.items() if not r.has_active_work()), None)
            if victim is not None:
                del self._runners[victim]
                logger.debug("Runner cache full; evicted %s", victim)

        module_path, class_name = _RUNNER_IMPORTS[kind]
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        runner = cls(
            loop=loop,
            simulator_class=simulator_class,
            output_folder=output_folder,
            max_parallel=max_parallel,
        )
        self._runners[key] = runner
        logger.debug(f"Created {class_name}: output={output_folder}")
        return runner

    def reset(self) -> None:
        """Force-invalidate all runners. Used by test fixtures."""
        self._runners.clear()
        self._loop = None

    def _get_existing(self, kind: str, simulator: str | None) -> Any | None:
        """Most-recently-used cached runner of ``kind``.

        ``simulator`` (a class name, e.g. ``"LTspiceWSL"``) narrows the match —
        pass the job's own ``simulator`` field when cancelling, so the kill uses
        that simulator's executable names and the launching runner's cancel
        events rather than whichever runner was used last.
        """
        of_kind = [(cls, runner) for (k, cls, _f), runner in self._runners.items() if k == kind]
        if simulator is not None:
            named = [runner for cls, runner in of_kind if cls.__name__ == simulator]
            if named:
                return named[-1]
            # Name matched nothing (runner evicted, or a recovered job whose
            # recorded name predates this session). With a single live runner
            # of this kind, prefer it over giving up: its kill is token-scoped,
            # so a class mismatch just matches no process — same as returning
            # None — while a match kills the right one.
            if len(of_kind) != 1:
                return None
        return of_kind[-1][1] if of_kind else None

    def get_batch_runner_for(self, job: Any) -> Any | None:
        """Runner to cancel batch ``job`` with.

        Batch cancel state (the cancel event that stops the submission loop)
        lives on the instance that launched the batch — with several runners
        of one kind cached (distinct output folders), most-recent is not
        necessarily the owner, so prefer the instance whose cancel registry
        owns the job id. Fall back to the most-recent runner of the kind for
        a job whose owner is gone (e.g. recovered after a restart): its
        token-scoped kill still reaches the right processes.
        """
        kind = "mc" if job.job_type == "montecarlo" else "sweep"
        for (k, _cls, _folder), runner in self._runners.items():
            if k == kind and runner.owns_batch_job(job.job_id):
                return runner
        return self._get_existing(kind, None)

    def get_existing_sim_runner(self, simulator: str | None = None) -> SimulationRunner | None:
        """Return a cached ``SimulationRunner`` if present (see ``_get_existing``)."""
        return self._get_existing("sim", simulator)

    def get_sim_runner(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> SimulationRunner:
        """Get or create a SimulationRunner."""
        return self._get_or_create("sim", loop, simulator_class, output_folder, max_parallel)

    def get_sweep_runner(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> SweepRunner:
        """Get or create a SweepRunner."""
        return self._get_or_create("sweep", loop, simulator_class, output_folder, max_parallel)

    def get_mc_runner(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> MonteCarloRunner:
        """Get or create a MonteCarloRunner."""
        return self._get_or_create("mc", loop, simulator_class, output_folder, max_parallel)
