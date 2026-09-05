"""Centralized runner lifecycle management.

Owns the experiment runner instances. Replaces the fragile module-level
singleton pattern where each tool module independently checked for staleness
(event loop, simulator, output folder), and keeps one invalidation mechanism
for all of them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ltspice_mcp.lib.runner_base import DEFAULT_MAX_PARALLEL

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

# Import paths for lazy loading (avoids circular imports with state.py)
_RUNNER_IMPORTS: dict[str, tuple[str, str]] = {
    "experiment": ("ltspice_mcp.lib.experiment_runner", "ExperimentRunner"),
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

    def get_experiment_runner_for(self, job: Any) -> ExperimentRunner | None:
        """Return the live coordinator that owns an experiment job."""
        for (kind, _cls, _folder), runner in self._runners.items():
            if kind == "experiment" and runner.owns_experiment_job(job.job_id):
                return runner
        return None

    def get_experiment_runner(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> ExperimentRunner:
        """Get or create an ExperimentRunner."""
        return self._get_or_create(
            "experiment",
            loop,
            simulator_class,
            output_folder,
            max_parallel,
        )
