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
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    LegacyJobRecord,
    legacy_record_message,
    legacy_record_observation,
)
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.lib.simulator import simulator_dialect
from ltspice_mcp.lib.store import Store

if TYPE_CHECKING:
    from mcp import types

    from ltspice_mcp.tools._base import RegisteredTool

logger = logging.getLogger(__name__)

# Cap on parsed-result (RawRead) cache entries. Each can pin a multi-MB raw, so
# a long-lived session querying many circuits must not retain them all; LRU
# eviction past this just re-parses on the next access.
RESULT_CACHE_MAXSIZE = 32

# Re-export the job-type surface so existing
# ``from ltspice_mcp.state import LegacyJobRecord`` imports keep working.
__all__ = [
    "NON_TERMINAL_LIVE_STATUSES",
    "TERMINAL_STATUSES",
    "ExperimentJob",
    "LegacyJobRecord",
    "SessionState",
    "legacy_record_message",
    "legacy_record_observation",
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
        runners: RunnerManager (sim/sweep/MC/experiment runner lifecycle)
        working_dir: Base directory for relative paths
        tool_defs / tool_dispatch / field_owners: Profile-filtered MCP tool exposure
        sweep_configs / mc_configs: Sweep and Monte Carlo run configurations
            held for the session, keyed by config_id
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
    diagnostics: list[str] = field(default_factory=list)
    """Startup diagnostics (bad simulator path, requested≠active fallback, WSL
    auto-detection). Logged at startup and carried verbatim on the ``inspect``
    capabilities payload, which is where a client can see the degradation —
    the log itself reaches nobody but whoever started the server."""
    _touched_recent: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths already recorded in the recent-circuits index this session."""
    config_write_attempted: bool = field(default=False, repr=False)
    """Whether the lazy default-config write has been tried this session (once)."""
    raw_dialect_hints: dict[Path, str | None] = field(default_factory=dict, repr=False)
    """Raw dialect per job-resolved raw path, recorded when the path is
    resolved (``services._resolve_result_file``) and read by ``load_raw`` —
    a per-run simulator override's raw must not parse with the session
    default's dialect. Paths never resolved through a job aren't listed."""
    client_log_level: str | None = field(default=None, repr=False)
    """Minimum log level the client requested via logging/setLevel, or None
    when the client never set one (send everything — the pre-setLevel
    default). Registering the setLevel handler is also what makes the SDK
    declare the logging capability in the initialize result."""

    @property
    def store(self) -> Store:
        """Every path this session writes. See ``lib/store.py`` for the layout.

        A value object over the working directory, so it is rebuilt per access
        rather than cached — nothing about it is stateful, and a session that
        changed its working directory would otherwise keep writing to the old
        one.
        """
        return Store(self.working_dir)

    @property
    def raw_dialect(self) -> str | None:
        """spicelib ``RawRead`` dialect for the default simulator.

        Returns ``None`` for LTspice (auto-detect works) and an explicit
        dialect string for simulators whose raw files lack the ``Command:``
        header that spicelib needs for auto-detection.
        """
        return simulator_dialect(self.default_simulator)

    # ------------------------------------------------------------------
    # Tool surface — built on FIRST ACCESS, not at session creation. The
    # library door (Api) calls handlers directly and never reads these, so it
    # never pays the tools-package import (mcp + the analysis chain); the MCP
    # server touches tool_defs during its handshake and builds then. A
    # property, not a flag: no caller can ever observe an empty surface.
    # ------------------------------------------------------------------

    @cached_property
    def _surface(
        self,
    ) -> "tuple[list[types.Tool], dict[str, RegisteredTool], dict[str, tuple[str, ...]]]":
        from ltspice_mcp.tools import get_tools
        from ltspice_mcp.tools._base import registry as tool_registry

        defs, dispatch = get_tools()
        owners = tool_registry.field_owners()
        return (defs, dispatch, owners)

    @property
    def tool_defs(self) -> "list[types.Tool]":
        """Profile-filtered advertised tool definitions."""
        return self._surface[0]

    @property
    def tool_dispatch(self) -> "dict[str, RegisteredTool]":
        """Tool name -> RegisteredTool dispatch map for the active profile."""
        return self._surface[1]

    @property
    def field_owners(self) -> "dict[str, tuple[str, ...]]":
        """Advertised top-level wire fields -> owning tool names."""
        return self._surface[2]

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
        on the session and logged at startup.
        """
        from ltspice_mcp.lib.simulator import select_default_simulator

        diagnostics = diagnostics if diagnostics is not None else []
        default = select_default_simulator(available, config, diagnostics)
        registry = JobRegistry(
            persist_enabled=config.persist_jobs,
            working_dir=config.working_dir,
        )

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
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Job-registry delegation (API preserved for all callers)
    # ------------------------------------------------------------------

    @property
    def legacy_records(self) -> MutableMapping[str, LegacyJobRecord]:
        """Type-filtered view of the job records earlier releases wrote."""
        return self.job_registry.legacy_records

    @property
    def experiment_jobs(self) -> MutableMapping[str, ExperimentJob]:
        """Type-filtered view of experiment coordinator jobs."""
        return self.job_registry.experiment_jobs

    @property
    def all_jobs(self) -> dict[str, "LegacyJobRecord | ExperimentJob"]:
        """The union job store — every job regardless of run type."""
        return self.job_registry.jobs

    def add_experiment_job(
        self,
        experiment_job: ExperimentJob,
        *,
        already_persisted: bool = False,
    ) -> None:
        self.job_registry.add_experiment_job(
            experiment_job,
            already_persisted=already_persisted,
        )

    def persist_job(self, job: "LegacyJobRecord | ExperimentJob") -> None:
        self.job_registry.persist_job(job)

    def ensure_jobs_loaded_for(self, circuit_path: Path) -> None:
        self.job_registry.ensure_loaded_for(circuit_path)

    async def ensure_jobs_loaded_for_async(self, circuit_path: Path) -> None:
        await self.job_registry.ensure_loaded_for_async(circuit_path)

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
        await self.job_registry.cancel_running(self.runners, self)
        await self.job_registry.drain_pending()
