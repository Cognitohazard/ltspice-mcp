"""Per-session container: config, simulators, caches, job registry.

The job domain types (``SimulationJob``, ``BatchJob``, ``SweepConfig``,
``SweepDimension``, ``MonteCarloConfig``) and status constants live in
``lib/job_types.py``; they're re-exported here so call sites that
imported them from ``state`` keep working. Splitting them out broke a
cluster of import cycles — see ``lib/job_types.py`` for the full story.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    MonteCarloConfig,
    SimulationJob,
    SweepConfig,
    SweepDimension,
)
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.lib.simulator import simulator_dialect

if TYPE_CHECKING:
    from ltspice_mcp.tools._base import RegisteredTool

logger = logging.getLogger(__name__)

# Re-export the job-type surface so existing
# ``from ltspice_mcp.state import SimulationJob`` imports keep working.
__all__ = [
    "NON_TERMINAL_LIVE_STATUSES",
    "TERMINAL_STATUSES",
    "BatchJob",
    "MonteCarloConfig",
    "SessionState",
    "SimulationJob",
    "SweepConfig",
    "SweepDimension",
]


@dataclass
class SessionState:
    """Per-session container: config, simulators, caches, job registry.

    Created at server startup and persists for the server lifetime. Job
    lifecycle (in-memory dicts, disk persistence, eviction, interrupted
    recovery) is delegated to ``JobRegistry`` in ``lib/job_registry.py``;
    ``state.jobs`` / ``state.add_job`` etc. are kept as thin delegators
    so call sites don't change.

    Attributes:
        config: Server configuration loaded from TOML/env vars
        available_simulators: Simulators detected at startup
        default_simulator: Simulator to use when not specified by user
        editors: Cache of parsed SpiceEditor instances
        results: Cache of parsed RawRead instances
        libraries: Loaded component libraries
        runners: RunnerManager (sim/sweep/MC runner lifecycle)
        working_dir: Base directory for relative paths
        tool_defs / tool_dispatch: Profile-filtered MCP tool exposure
        sweep_configs / mc_configs: Saved configs keyed by config_id
        job_registry: Owns sim_jobs / batch_jobs + disk persistence
    """

    config: ServerConfig
    available_simulators: dict[str, type]
    default_simulator: type | None
    editors: FileCache
    results: FileCache
    libraries: LibraryManager
    runners: RunnerManager
    working_dir: Path
    job_registry: JobRegistry = field(default_factory=lambda: JobRegistry(persist_enabled=False))
    tool_defs: list[types.Tool] = field(default_factory=list)
    tool_dispatch: dict[str, "RegisteredTool"] = field(default_factory=dict)
    sweep_configs: dict[str, SweepConfig] = field(default_factory=dict)
    mc_configs: dict[str, MonteCarloConfig] = field(default_factory=dict)
    _touched_recent: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths already recorded in the recent-circuits index this session."""

    @property
    def raw_dialect(self) -> str | None:
        """spicelib ``RawRead`` dialect for the default simulator.

        Returns ``None`` for LTspice (auto-detect works) and an explicit
        dialect string for simulators whose raw files lack the ``Command:``
        header that spicelib needs for auto-detection.
        """
        return simulator_dialect(self.default_simulator)

    @classmethod
    def create(cls, config: ServerConfig, available: dict[str, type]) -> "SessionState":
        """Factory method to create session state at server startup."""
        from ltspice_mcp.lib.simulator import select_default_simulator
        from ltspice_mcp.tools import get_tools_for_profile

        default = select_default_simulator(available, config)
        tool_defs, tool_dispatch = get_tools_for_profile(config.tool_profile)
        registry = JobRegistry(persist_enabled=config.persist_jobs)

        return cls(
            config=config,
            available_simulators=available,
            default_simulator=default,
            editors=FileCache(),
            results=FileCache(),
            libraries=LibraryManager(available),
            runners=RunnerManager(),
            working_dir=config.working_dir,
            job_registry=registry,
            tool_defs=tool_defs,
            tool_dispatch=tool_dispatch,
        )

    # ------------------------------------------------------------------
    # Job-registry delegation (API preserved for all callers)
    # ------------------------------------------------------------------

    @property
    def jobs(self) -> dict[str, SimulationJob]:
        return self.job_registry.sim_jobs

    @property
    def batch_jobs(self) -> dict[str, BatchJob]:
        return self.job_registry.batch_jobs

    def add_job(self, job: SimulationJob) -> None:
        self.job_registry.add_sim_job(job)

    def add_batch_job(self, batch_job: BatchJob) -> None:
        self.job_registry.add_batch_job(batch_job)

    def persist_job(self, job: "SimulationJob | BatchJob") -> None:
        self.job_registry.persist_job(job)

    def persist_batch_progress(self, batch_job: BatchJob) -> None:
        self.job_registry.persist_batch_progress(batch_job)

    def ensure_jobs_loaded_for(self, circuit_path: Path) -> None:
        self.job_registry.ensure_loaded_for(circuit_path)

    # ------------------------------------------------------------------
    # Recent-circuits index (session-scoped state, not job-scoped)
    # ------------------------------------------------------------------

    def note_recent_circuit(self, resolved_path: Path) -> None:
        """Record a circuit in the global recent-circuits index, once per session.

        ``resolved_path`` must already be a resolved, sandbox-validated path.
        The per-session debounce prevents rewriting ``recent.json`` on every
        tool call that touches the same circuit.
        """
        if not self.config.persist_jobs:
            return
        if resolved_path in self._touched_recent:
            return
        self._touched_recent.add(resolved_path)
        try:
            from ltspice_mcp.lib import recent

            recent.touch(resolved_path)
        except Exception as e:
            logger.debug("recent.touch(%s) failed: %s", resolved_path, e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Clean up session resources at server shutdown."""
        self.editors.clear()
        self.results.clear()
        await self.job_registry.cancel_running(self.runners, self)
        await self.job_registry.drain_pending()
