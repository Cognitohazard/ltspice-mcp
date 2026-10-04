"""Per-session container: config, simulators, caches, job registry.

The job status constants live in ``lib/job_types.py``; they're re-exported
here so call sites can read them from either place. Splitting them out broke
a cluster of import cycles — see ``lib/job_types.py`` for the full story.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from ltspice_mcp.config import SANDBOX_ENV, SANDBOX_KEY, SANDBOX_SECTION, ServerConfig
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES, TERMINAL_STATUSES
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.result_cache import ResultCache
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.lib.simulator import simulator_dialect
from ltspice_mcp.lib.store import Store

if TYPE_CHECKING:
    from mcp import types

    from ltspice_mcp.tools._base import RegisteredTool
    from ltspice_mcp.tools.run_code import CodeWorker

logger = logging.getLogger(__name__)

# Re-export the job-status vocabulary so a caller can read it from either
# ``ltspice_mcp.state`` or ``ltspice_mcp.lib.job_types``.
__all__ = [
    "NON_TERMINAL_LIVE_STATUSES",
    "TERMINAL_STATUSES",
    "ExperimentJob",
    "SessionState",
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
        available_simulators: Simulators detected at startup, by family
        named_simulators: ``[simulator.executables]`` bound at startup, by
            selector (``"ltspice:xvii"``); each its own simulator class
            (``simulator.bind_named_executable``)
        default_simulator: Simulator to use when not specified by user
        editors: Cache of parsed SpiceEditor instances
        results: Bounded content cache of resident decoded results
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
    results: ResultCache
    libraries: LibraryManager
    runners: RunnerManager
    working_dir: Path
    job_registry: JobRegistry = field(default_factory=lambda: JobRegistry(persist_enabled=False))
    named_simulators: dict[str, type] = field(default_factory=dict)
    sandbox_pinned: bool = False
    """The sandbox was given explicitly when the session was opened
    (``Api(allowed_paths=...)``). That outranks the file at startup, so it keeps
    outranking it: the sandbox does not follow the file for this session."""
    _sandbox: tuple[tuple[int, int] | None, list[Path]] = field(
        init=False, repr=False, compare=False
    )
    """The config file's (mtime_ns, size) stamp and the sandbox read at that
    stamp. One tuple, replaced in one assignment: resource reads and some path
    resolutions run on worker threads, and a reader must never pair a new
    stamp with an old list — nor wait on another thread's reload to find out."""
    diagnostics: list[str] = field(default_factory=list)
    """Startup diagnostics (bad simulator path, requested≠active fallback, WSL
    auto-detection). Logged at startup and carried verbatim on the ``inspect``
    capabilities payload, which is where a client can see the degradation —
    the log itself reaches nobody but whoever started the server."""
    _touched_recent: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths already recorded in the recent-circuits index this session."""
    config_write_attempted: bool = field(default=False, repr=False)
    """Whether the lazy default-config write has been tried this session (once)."""
    guide_read: bool = field(default=False, repr=False)
    """Whether this session has read the guide, through an ``inspect`` guide
    query or a ``spice://guide`` resource. Until it has, the first tool reply
    carries one reminder to read the core (``server.call_tool``)."""
    guide_reminded: bool = field(default=False, repr=False)
    """Whether that one reminder has been sent."""
    code_worker: "CodeWorker | None" = field(default=None, repr=False)
    """The ``run_code`` worker supervisor, created on the first call and
    closed at shutdown."""

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
        """Recorded dialect of the default producing simulator, when known."""
        return simulator_dialect(self.default_simulator)

    # ------------------------------------------------------------------
    # Tool surface — built on FIRST ACCESS, not at session creation. The
    # Python API (Api) calls handlers directly and never reads these, so it
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

        # A registration may declare a config gate; the surface is what the
        # gates leave open for this session's config.
        defs, dispatch = get_tools(self.config.tool_listing, config=self.config)
        owners = {
            name: kept
            for name, tools in tool_registry.field_owners().items()
            if (kept := tuple(owner for owner in tools if owner in dispatch))
        }
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

    def __post_init__(self) -> None:
        self._sandbox = (_file_stamp(self.config.config_path), self.config.allowed_paths)

    def allowed_paths(self) -> list[Path]:
        """The sandbox, re-read from the config file whenever that file changed.

        The refusal an agent gets names the config line that widens the sandbox;
        picking the edit up on the next call is what makes that line the agent's
        own to act on. Only ``[security] allowed_paths`` follows the file: the
        rest of it is startup state (detected simulators, runners, caches). A
        pinned sandbox does not follow it at all; see ``sandbox_pinned``.

        Every reader of the sandbox calls this: ``config.allowed_paths`` is
        the list the session opened with, and a report or a resolution made
        after an edit must see the edit. Two threads that notice the same edit
        both reload it, which is harmless; neither blocks the other.
        """
        if self.sandbox_pinned:
            return self.config.allowed_paths
        stamp = _file_stamp(self.config.config_path)
        seen, paths = self._sandbox
        if stamp != seen:
            paths = ServerConfig.load(
                self.config.config_path, overrides={"working_dir": self.config.working_dir}
            ).allowed_paths
            self._sandbox = (stamp, paths)
        return paths

    def sandbox_guidance(self) -> str:
        """What a caller refused by the sandbox can do about it.

        An agent cannot widen the sandbox itself except through the setting
        that holds it, so a refusal that does not name that setting (and the
        move-the-file fallback) dead-ends. Every surface that reports a refusal
        carries this text: a tool's structured ``hint``, a failed resource
        read, and a note on the exception the Python API raises. One builder,
        so they cannot drift.
        """
        allowed = ", ".join(str(p) for p in self.allowed_paths())
        config_path = self.config.config_path
        key = f"[{SANDBOX_SECTION}] {SANDBOX_KEY}"
        # The branches follow the loader's precedence: an explicit argument, then
        # the environment, then the file.
        if self.sandbox_pinned:
            widen = (
                f"open a new Api with its directory added to {SANDBOX_KEY}: this "
                f"session's sandbox is the Api({SANDBOX_KEY}=...) it was opened with, "
                f"which replaces {key} in {config_path} for the whole session."
            )
        elif os.environ.get(SANDBOX_ENV):
            widen = (
                f"widen {SANDBOX_ENV}, which is set in the server's environment and "
                f"replaces {key} in {config_path} (restart required)."
            )
        else:
            widen = (
                f"add its directory to {key} in {config_path} — that file is re-read "
                f"on the next call, no restart. {SANDBOX_ENV} sets the same list "
                "and overrides the file (restart required)."
            )
        return (
            f"Allowed paths: {allowed}\n"
            "To work on this file: pass its content inline where the argument takes "
            "text (a compare reference), copy it into one of those directories, or "
            f"{widen} An inspect capabilities query shows the full sandbox configuration."
        )

    @classmethod
    def create(
        cls,
        config: ServerConfig,
        available: dict[str, type],
        diagnostics: list[str] | None = None,
        *,
        named: dict[str, type] | None = None,
        sandbox_pinned: bool = False,
    ) -> "SessionState":
        """Factory method to create session state at server startup.

        ``diagnostics`` carries any startup notes accumulated during simulator
        detection (e.g. a bad configured path); ``select_default_simulator``
        appends to it when it has to fall back, and the merged list is stored
        on the session and logged at startup. ``named`` is the named
        executables ``detect_named_simulators`` bound. ``sandbox_pinned`` says
        the caller gave ``allowed_paths`` explicitly (see the field).
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
            # eviction would drop. Resident results have byte and entry bounds.
            editors=FileCache(),
            results=ResultCache(),
            libraries=LibraryManager(available),
            runners=RunnerManager(),
            working_dir=config.working_dir,
            job_registry=registry,
            named_simulators=dict(named or {}),
            diagnostics=diagnostics,
            sandbox_pinned=sandbox_pinned,
        )

    # ------------------------------------------------------------------
    # Job-registry delegation (API preserved for all callers)
    # ------------------------------------------------------------------

    @property
    def all_jobs(self) -> dict[str, ExperimentJob]:
        """The job store — every job this session knows."""
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

    def persist_job(self, job: ExperimentJob) -> None:
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
        if self.code_worker is not None:
            await self.code_worker.close()
        await self.job_registry.cancel_running(self.runners, self)
        await self.job_registry.drain_pending()


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)
