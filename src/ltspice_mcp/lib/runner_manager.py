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


class RunnerManager:
    """Creates and caches runner instances, invalidating when context changes.

    A runner is stale when any of these change:
    - The asyncio event loop (e.g., between test runs)
    - The simulator class (e.g., user switches from LTspice to ngspice)
    - The output folder (e.g., WSL temp dir changes)
    """

    def __init__(self) -> None:
        self._runners: dict[str, Any] = {}

        # Track the context that runners were created with
        self._loop: asyncio.AbstractEventLoop | None = None
        self._simulator_class: type | None = None
        self._output_folder: Path | None = None

    def _ensure_fresh(
        self, loop: asyncio.AbstractEventLoop, simulator_class: type, output_folder: Path
    ) -> None:
        """Invalidate all runners if context has changed."""
        if (
            self._loop is None
            or self._loop is not loop
            or self._simulator_class is not simulator_class
            or self._output_folder != output_folder
        ):
            self._runners.clear()
            self._loop = loop
            self._simulator_class = simulator_class
            self._output_folder = output_folder

    def _get_or_create(
        self,
        key: str,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int,
    ) -> Any:
        """Get a cached runner or create a new one."""
        self._ensure_fresh(loop, simulator_class, output_folder)

        if key not in self._runners:
            module_path, class_name = _RUNNER_IMPORTS[key]
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            self._runners[key] = cls(
                loop=loop,
                simulator_class=simulator_class,
                output_folder=output_folder,
                max_parallel=max_parallel,
            )
            logger.debug(f"Created {class_name}: output={output_folder}")
        else:
            # A cached runner keeps the max_parallel it was created with, but a
            # later run_sweep/run_montecarlo may request a different cap. Update
            # it in place: each batch rebuilds its spicelib SimRunner from
            # ``self.max_parallel`` at launch, so the new cap takes effect on the
            # next batch this runner starts. Updating the attribute (vs. recreating
            # the instance) preserves the per-job cancel-event / live-process map
            # that an in-flight batch — and cancel_job — depend on.
            self._runners[key].max_parallel = max_parallel

        return self._runners[key]

    def reset(self) -> None:
        """Force-invalidate all runners. Used by test fixtures."""
        self._runners.clear()
        self._loop = None
        self._simulator_class = None
        self._output_folder = None

    def get_existing_sim_runner(self) -> SimulationRunner | None:
        """Return the currently cached ``SimulationRunner`` if present."""
        return self._runners.get("sim")

    def get_existing_sweep_runner(self) -> SweepRunner | None:
        """Return the currently cached ``SweepRunner`` if present."""
        return self._runners.get("sweep")

    def get_existing_mc_runner(self) -> MonteCarloRunner | None:
        """Return the currently cached ``MonteCarloRunner`` if present."""
        return self._runners.get("mc")

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
