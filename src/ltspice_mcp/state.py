"""Per-session container: config, simulators, caches, job registry.

The job domain types (``SimulationJob``, ``BatchJob``, ``SweepConfig``,
``SweepDimension``, ``MonteCarloConfig``) and status constants live in
``lib/job_types.py``; they're re-exported here so call sites that
imported them from ``state`` keep working. Splitting them out broke a
cluster of import cycles — see ``lib/job_types.py`` for the full story.
"""

import asyncio
import logging
from collections.abc import MutableMapping
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
    RunRef,
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

# Cap on parsed-result (RawRead) cache entries. Each can pin a multi-MB raw, so
# a long-lived session querying many circuits must not retain them all; LRU
# eviction past this just re-parses on the next access.
RESULT_CACHE_MAXSIZE = 32

# Re-export the job-type surface so existing
# ``from ltspice_mcp.state import SimulationJob`` imports keep working.
__all__ = [
    "NON_TERMINAL_LIVE_STATUSES",
    "TERMINAL_STATUSES",
    "BatchJob",
    "MonteCarloConfig",
    "RunRef",
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
        job_registry: Owns the union job store + disk persistence
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
    diagnostics: list[str] = field(default_factory=list)
    """Startup diagnostics (bad simulator path, requested≠active fallback, WSL
    auto-detection). Surfaced via ``server_status`` so silent degradation is
    visible to the client instead of buried in the server log."""
    _touched_recent: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths already recorded in the recent-circuits index this session."""
    config_write_attempted: bool = field(default=False, repr=False)
    """Whether the lazy default-config write has been tried this session (once)."""
    asc_snapshots: dict[str, bytes] = field(default_factory=dict, repr=False)
    """Pre-first-edit byte snapshots of .asc schematics touched this session,
    keyed by resolved path string. Captured before the first in-session
    mutation; backs ``reset_schematic`` (revert to last good state)."""

    @property
    def raw_dialect(self) -> str | None:
        """spicelib ``RawRead`` dialect for the default simulator.

        Returns ``None`` for LTspice (auto-detect works) and an explicit
        dialect string for simulators whose raw files lack the ``Command:``
        header that spicelib needs for auto-detection.
        """
        return simulator_dialect(self.default_simulator)

    @classmethod
    def create(
        cls,
        config: ServerConfig,
        available: dict[str, type],
        diagnostics: list[str] | None = None,
    ) -> "SessionState":
        """Factory method to create session state at server startup.

        ``diagnostics`` carries any startup notes accumulated during simulator
        detection (e.g. a bad configured path); ``select_default_simulator``
        appends to it when it has to fall back, and the merged list is stored
        on the session for ``server_status`` to surface.
        """
        from ltspice_mcp.lib.simulator import select_default_simulator
        from ltspice_mcp.tools import get_tools_for_profile

        diagnostics = diagnostics if diagnostics is not None else []
        default = select_default_simulator(available, config, diagnostics)
        tool_defs, tool_dispatch = get_tools_for_profile(config.tool_profile)
        registry = JobRegistry(persist_enabled=config.persist_jobs)

        return cls(
            config=config,
            available_simulators=available,
            default_simulator=default,
            # Editors are unbounded: they may hold unsaved in-memory edits that
            # eviction would drop. Results are immutable parsed RawReads, safe to
            # LRU-evict so a long session over many circuits doesn't grow without
            # bound (each can pin a multi-MB raw).
            editors=FileCache(),
            results=FileCache(maxsize=RESULT_CACHE_MAXSIZE),
            libraries=LibraryManager(available),
            runners=RunnerManager(),
            working_dir=config.working_dir,
            job_registry=registry,
            tool_defs=tool_defs,
            tool_dispatch=tool_dispatch,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Job-registry delegation (API preserved for all callers)
    # ------------------------------------------------------------------

    @property
    def jobs(self) -> MutableMapping[str, SimulationJob]:
        """Type-filtered view of the single-simulation jobs."""
        return self.job_registry.sim_jobs

    @property
    def batch_jobs(self) -> MutableMapping[str, BatchJob]:
        """Type-filtered view of the batch (sweep/MC) jobs."""
        return self.job_registry.batch_jobs

    @property
    def all_jobs(self) -> dict[str, "SimulationJob | BatchJob"]:
        """The union job store — every job regardless of run type."""
        return self.job_registry.jobs

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

    async def note_recent_circuit(self, resolved_path: Path) -> None:
        """Record a circuit in the global recent-circuits index, once per session.

        ``resolved_path`` must already be a resolved, sandbox-validated path.
        The per-session debounce prevents rewriting ``recent.json`` on every
        tool call that touches the same circuit.

        The write itself runs in a worker thread: ``recent.touch`` polls a
        cross-process file lock (up to 10 s with ``time.sleep``) and does a
        durable double-fsync write — both would stall every concurrent
        request if run on the event loop. The debounce set is updated before
        the await, so a cancelled caller cannot double-write.
        """
        if not self.config.persist_jobs:
            return
        if resolved_path in self._touched_recent:
            return
        self._touched_recent.add(resolved_path)
        try:
            from ltspice_mcp.lib import recent

            await asyncio.to_thread(recent.touch, resolved_path)
        except Exception as e:
            logger.debug("recent.touch(%s) failed: %s", resolved_path, e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Clean up session resources at server shutdown."""
        self.editors.clear()
        self.results.clear()
        self.asc_snapshots.clear()
        await self.job_registry.cancel_running(self.runners, self)
        await self.job_registry.drain_pending()
