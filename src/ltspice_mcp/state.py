"""Session state management and simulation job tracking."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.runner_manager import RunnerManager

if TYPE_CHECKING:
    from ltspice_mcp.tools._base import RegisteredTool

logger = logging.getLogger(__name__)

# Terminal job statuses — jobs in these states are eligible for eviction.
# "interrupted" is terminal because the owning runner is gone; metadata and
# any partial outputs are preserved but the job cannot resume in-process.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "timeout", "cancelled", "interrupted"}
)

# Statuses that only make sense while a runner owns the job. Seeing one
# in a persisted record means the prior server died mid-run.
NON_TERMINAL_LIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})

# Maximum finished jobs to retain per dict (jobs, batch_jobs).
_MAX_FINISHED_JOBS = 200

# LTspice .raw header magic. Classic files start with ASCII ``Title:``;
# newer LTspice writes a UTF-16 LE BOM followed by the same ``Title:``.
_RAW_HEADER_ASCII = b"Title:"
_RAW_HEADER_UTF16 = b"\xff\xfeT\x00i\x00t\x00l\x00e\x00:\x00"


def _has_valid_raw(path: Path | None) -> bool:
    """True if ``path`` looks like a real LTspice ``.raw`` file.

    Checks the header magic so a truncated or unrelated file at the same
    path doesn't mis-promote an ``interrupted`` job to ``completed``.
    """
    if path is None:
        return False
    try:
        with path.open("rb") as f:
            header = f.read(len(_RAW_HEADER_UTF16))
    except OSError:
        return False
    return header.startswith(_RAW_HEADER_ASCII) or header.startswith(_RAW_HEADER_UTF16)


@dataclass
class SweepDimension:
    """One axis of a parameter sweep.

    Attributes:
        type: "component" (add_value_sweep) or "parameter" (add_param_sweep)
        name: Component reference (e.g. "R1") or parameter name (e.g. "TEMP")
        start: Start value for sweep range
        stop: Stop value for sweep range
        step: Step size — mutually exclusive with points
        points: Number of points — mutually exclusive with step
        scale: "linear" or "log"
    """

    type: Literal["component", "parameter"]
    name: str
    start: float
    stop: float
    step: float | None = None
    points: int | None = None
    scale: str = "linear"


@dataclass
class SweepConfig:
    """Configuration for a multi-dimensional parameter sweep.

    Attributes:
        netlist: Path to the netlist to sweep (bound at config creation)
        dimensions: List of sweep axes (one per varied parameter)
    """

    netlist: Path
    dimensions: list[SweepDimension] = field(default_factory=list)


@dataclass
class MonteCarloConfig:
    """Configuration for a Monte Carlo analysis run.

    Attributes:
        netlist: Path to the netlist
        type_tolerances: Per-component-type tolerances: prefix -> (tolerance, distribution)
            e.g. {"R": (0.05, "uniform")} means all resistors get 5% uniform tolerance
        component_overrides: Per-component tolerances: ref -> (tolerance, distribution)
            e.g. {"R1": (0.01, "normal")} overrides R1 with 1% normal distribution
        num_runs: Number of Monte Carlo runs (default 100)
    """

    netlist: Path
    type_tolerances: dict[str, tuple[float, str]] = field(default_factory=dict)
    component_overrides: dict[str, tuple[float, str]] = field(default_factory=dict)
    num_runs: int = 100


@dataclass
class BatchJob:
    """Track state of a running or completed batch simulation job.

    Attributes:
        job_id: Unique identifier for this batch job
        job_type: "sweep" or "montecarlo"
        netlist: Path to the netlist file being processed
        total_runs: Total number of runs in this batch
        completed_runs: Number of runs completed so far
        failed_runs: Number of runs that failed
        status: Current job status
        started_at: When the batch job started
        completed_at: When the batch job finished (None if still running)
        error: Error message if the whole job failed
        done_event: Event signaled when batch completes or is cancelled
        run_results: Per-run results: run_index -> {raw_file, log_file, params}
        sweep_config: SweepConfig stored for reference during execution
        mc_config: MonteCarloConfig stored for reference during execution
    """

    job_id: str
    job_type: Literal["sweep", "montecarlo"]
    netlist: Path
    total_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    status: Literal["running", "completed", "failed", "cancelled", "interrupted"] = "running"
    started_at: datetime = field(default_factory=now)
    completed_at: datetime | None = None
    error: str | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    run_results: dict[int, dict] = field(default_factory=dict)
    sweep_config: SweepConfig | None = None
    mc_config: MonteCarloConfig | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class SimulationJob:
    """Track state of a running or completed simulation.

    Attributes:
        job_id: Unique identifier for this job
        netlist: Path to the netlist file being simulated
        simulator: Name of simulator used (ltspice, ngspice, etc.)
        status: Current job status
        started_at: When simulation started
        completed_at: When simulation finished (None if still running)
        raw_file: Path to generated .raw file (None until simulation completes)
        log_file: Path to simulation log file (None until available)
        error: Error message if simulation failed
        task: RunTask from spicelib (internal state, type: Any to avoid Phase 1 import)
        done_event: Event signaled when simulation completes
    """

    job_id: str
    netlist: Path
    simulator: str
    status: Literal[
        "queued", "running", "completed", "failed", "timeout", "cancelled", "interrupted"
    ]
    started_at: datetime
    completed_at: datetime | None = None
    raw_file: Path | None = None
    log_file: Path | None = None
    error: str | None = None
    task: Any | None = None  # RunTask from spicelib - typed as Any to defer Phase 3 import
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionState:
    """Global server state for the current session.

    Holds configuration, detected simulators, file caches, and active jobs.
    Created at server startup and persists for the server lifetime.

    Attributes:
        config: Server configuration loaded from TOML/env vars
        available_simulators: Simulators detected at startup
        default_simulator: Simulator to use when not specified by user
        editors: Cache of parsed SpiceEditor instances (FileCache[SpiceEditor])
        results: Cache of parsed RawRead instances (FileCache[RawRead])
        jobs: Active and completed simulation jobs by job_id
        libraries: Loaded component libraries
        working_dir: Base directory for relative paths
        sweep_configs: Stored sweep configurations keyed by config_id
        mc_configs: Stored Monte Carlo configurations keyed by config_id
        batch_jobs: Active and completed batch jobs keyed by job_id
    """

    config: ServerConfig
    available_simulators: dict[str, type]
    default_simulator: type | None
    editors: FileCache  # FileCache[SpiceEditor] - type parameter for documentation
    results: FileCache  # FileCache[RawRead]
    jobs: dict[str, SimulationJob]
    libraries: LibraryManager
    runners: RunnerManager
    working_dir: Path
    tool_defs: list[types.Tool] = field(default_factory=list)
    """MCP tool definitions filtered by the active tool profile."""
    tool_dispatch: dict[str, "RegisteredTool"] = field(default_factory=dict)
    """Tool name → registered tool metadata, filtered by the active tool profile."""
    sweep_configs: dict[str, SweepConfig] = field(default_factory=dict)
    mc_configs: dict[str, MonteCarloConfig] = field(default_factory=dict)
    batch_jobs: dict[str, BatchJob] = field(default_factory=dict)
    _loaded_circuits: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths whose persisted jobs have been loaded this session."""
    _touched_recent: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths already recorded in the recent-circuits index this session."""
    _pending_persist: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    """In-flight persistence writes; drained on shutdown."""
    _persist_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    """Per-job-id locks so successive writes for the same job serialise correctly.

    Entries are cleared in ``_evict_from`` when a job is removed from memory —
    removing them inside the write path would open a window where a new
    writer allocates a fresh Lock while an existing holder still owns the
    old one, defeating serialisation."""

    @classmethod
    def create(cls, config: ServerConfig, available: dict[str, type]) -> "SessionState":
        """Factory method to create session state at server startup.

        This is called by the server lifespan context manager to initialize
        the session state with detected simulators and empty caches.

        Args:
            config: Server configuration from ServerConfig.load()
            available: Available simulators from detect_simulators()

        Returns:
            Initialized SessionState instance
        """
        from ltspice_mcp.lib.simulator import select_default_simulator
        from ltspice_mcp.tools import get_tools_for_profile

        default = select_default_simulator(available, config)
        tool_defs, tool_dispatch = get_tools_for_profile(config.tool_profile)

        return cls(
            config=config,
            available_simulators=available,
            default_simulator=default,
            editors=FileCache(),
            results=FileCache(),
            jobs={},
            libraries=LibraryManager(available),
            runners=RunnerManager(),
            working_dir=config.working_dir,
            tool_defs=tool_defs,
            tool_dispatch=tool_dispatch,
        )

    def _evict_from(self, jobs_dict: dict) -> None:
        """Evict oldest terminal jobs from a single dict when over the limit.

        When ``config.persist_jobs`` is set, the on-disk record is deleted
        alongside the in-memory entry so the two never drift. Any per-job
        persistence lock is dropped here — it's safe once the job is out
        of the dict because no new ``persist_job`` calls can target it.
        """
        finished = [
            (jid, j)
            for jid, j in jobs_dict.items()
            if j.status in TERMINAL_STATUSES
        ]
        overflow = len(finished) - _MAX_FINISHED_JOBS
        if overflow <= 0:
            return
        finished.sort(key=lambda pair: pair[1].started_at)
        for jid, j in finished[:overflow]:
            del jobs_dict[jid]
            self._delete_persisted(j)
            self._persist_locks.pop(jid, None)

    def add_job(self, job: SimulationJob) -> None:
        """Register a simulation job and evict old finished jobs if needed."""
        self.jobs[job.job_id] = job
        self._evict_from(self.jobs)
        self.persist_job(job)

    def add_batch_job(self, batch_job: BatchJob) -> None:
        """Register a batch job and evict old finished batch jobs if needed."""
        self.batch_jobs[batch_job.job_id] = batch_job
        self._evict_from(self.batch_jobs)
        self.persist_job(batch_job)

    def persist_job(self, job: "SimulationJob | BatchJob") -> None:
        """Write a job's current state to its per-circuit sidecar file.

        When called from an asyncio event loop, the file IO is scheduled on
        a worker thread so the loop doesn't stall on slow filesystems (WSL
        cross-filesystem, network mounts). Successive writes for the same
        ``job_id`` are serialised through ``_persist_locks`` so on-disk
        order matches call order — a "completed" write after "running"
        always wins.
        """
        if not self.config.persist_jobs:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not in an event loop (tests, CLI usage) — write synchronously.
            self._persist_sync(job)
            return
        task = loop.create_task(self._persist_async(job))
        self._pending_persist.add(task)
        task.add_done_callback(self._pending_persist.discard)

    async def _persist_async(self, job: "SimulationJob | BatchJob") -> None:
        """Serialise writes for a single job id; swallow and log failures."""
        lock = self._persist_locks.get(job.job_id)
        if lock is None:
            lock = self._persist_locks.setdefault(job.job_id, asyncio.Lock())
        async with lock:
            await asyncio.to_thread(self._persist_sync, job)

    def _persist_sync(self, job: "SimulationJob | BatchJob") -> None:
        try:
            from ltspice_mcp.lib import job_store

            job_store.save_job(job)
        except Exception as e:
            # Persistence failures must never break simulation flow.
            logger.warning("Failed to persist job %s: %s", job.job_id, e)

    def persist_batch_progress(self, batch_job: "BatchJob") -> None:
        """Persist a batch job's in-progress state, throttled by run count.

        Per-run callbacks for sweeps and Monte Carlo can fire thousands of
        times per job; serialising the full ``run_results`` dict on each
        call is O(N²). Write only on a sparse schedule so crash-recovery
        sees near-current state without paying the quadratic IO cost.
        """
        if not self.config.persist_jobs:
            return
        total = batch_job.total_runs
        done = batch_job.completed_runs
        # Checkpoint ~20 times per batch plus always on the final run.
        step = max(1, total // 20) if total else 1
        if done == total or done % step == 0:
            self.persist_job(batch_job)

    def _delete_persisted(self, job: "SimulationJob | BatchJob") -> None:
        """Remove a job's on-disk record (used on eviction)."""
        if not self.config.persist_jobs:
            return
        try:
            from ltspice_mcp.lib import job_store

            job_store.delete_job(job)
        except Exception as e:
            logger.debug("Failed to delete persisted job %s: %s", job.job_id, e)

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

    def ensure_jobs_loaded_for(self, circuit_path: Path) -> None:
        """Load any persisted jobs for this circuit into memory, once per session.

        No-op when persistence is disabled, the path is not a circuit file,
        or the sidecar directory doesn't exist. Jobs in non-terminal states
        at load time are marked ``interrupted`` (their owning server is gone).
        """
        if not self.config.persist_jobs:
            return
        try:
            resolved = circuit_path.resolve()
        except OSError:
            return
        if resolved in self._loaded_circuits:
            return
        self._loaded_circuits.add(resolved)

        try:
            from ltspice_mcp.lib import job_store

            sim_jobs, batch_jobs = job_store.load_jobs_for_circuit(resolved)
        except Exception as e:
            logger.warning("Failed to load persisted jobs for %s: %s", resolved, e)
            return

        for sj in sim_jobs:
            if sj.job_id in self.jobs:
                continue
            self.jobs[sj.job_id] = sj
            # If the sim outputs exist on disk, the job may have finished
            # just before the crash — promote interrupted → completed.
            if sj.status == "interrupted" and _has_valid_raw(sj.raw_file):
                sj.status = "completed"
                sj.error = None
                self.persist_job(sj)
        for bj in batch_jobs:
            if bj.job_id in self.batch_jobs:
                continue
            self.batch_jobs[bj.job_id] = bj

    async def shutdown(self) -> None:
        """Clean up session resources at server shutdown.

        Clears file caches and cancels running jobs for graceful shutdown.
        """
        self.editors.clear()
        self.results.clear()
        # Cancel any running simulation jobs
        sim_runner = self.runners.get_existing_sim_runner()
        for job in self.jobs.values():
            if job.status in ("running", "queued"):
                if sim_runner is not None:
                    await sim_runner.cancel(job, self)
                else:
                    job.status = "cancelled"
                    job.completed_at = now()
                    job.done_event.set()
                    self.persist_job(job)
        # Cancel any running batch jobs
        sweep_runner = self.runners.get_existing_sweep_runner()
        mc_runner = self.runners.get_existing_mc_runner()
        for batch_job in self.batch_jobs.values():
            if batch_job.status == "running":
                if batch_job.job_type == "sweep" and sweep_runner is not None:
                    await sweep_runner.cancel(batch_job, self)
                elif batch_job.job_type == "montecarlo" and mc_runner is not None:
                    await mc_runner.cancel(batch_job, self)
                else:
                    batch_job.status = "cancelled"
                    batch_job.completed_at = now()
                    batch_job.done_event.set()
                    self.persist_job(batch_job)
            # Ensure the background task is awaited so exceptions aren't lost
            if batch_job.task is not None and not batch_job.task.done():
                batch_job.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await batch_job.task

        # Drain any pending persistence writes so the on-disk state
        # reflects this session's final mutations before we return.
        if self._pending_persist:
            await asyncio.gather(*self._pending_persist, return_exceptions=True)
