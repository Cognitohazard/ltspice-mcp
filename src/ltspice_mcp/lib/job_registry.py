"""In-memory registry for simulation and batch jobs.

Owns the ``sim_jobs`` / ``batch_jobs`` dicts plus all disk-persistence
coordination (sidecar writes, eviction, interrupted-job recovery). Split
out of ``SessionState`` so the per-session container stays focused on
simulator catalog, caches, and configuration.

``SessionState`` delegates its job-facing API to this class; call sites
continue to use ``state.jobs``, ``state.add_job``, etc.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ltspice_mcp.lib.job_lifecycle import recover, transition
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    SimulationJob,
)
from ltspice_mcp.lib.observability import emit_job_event

logger = logging.getLogger(__name__)

# Maximum finished jobs to retain per dict (sim_jobs, batch_jobs).
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
class JobRegistry:
    """Tracks simulation and batch jobs with optional disk persistence.

    Attributes:
        persist_enabled: When True, sidecar files are written alongside
            circuits and evictions delete them. When False, the registry
            behaves as pure in-memory dicts.
        sim_jobs: job_id -> SimulationJob
        batch_jobs: job_id -> BatchJob
    """

    persist_enabled: bool
    sim_jobs: dict[str, SimulationJob] = field(default_factory=dict)
    batch_jobs: dict[str, BatchJob] = field(default_factory=dict)
    _loaded_circuits: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths whose persisted jobs have been loaded this session."""
    _pending_persist: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    """In-flight persistence writes; drained on shutdown."""
    _persist_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    """Per-job-id locks serialising successive writes.

    Cleared in ``_evict_from`` when a job is removed from memory — removing
    them inside the write path would open a window where a new writer
    allocates a fresh Lock while an existing holder still owns the old
    one, defeating serialisation.
    """

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_sim_job(self, job: SimulationJob) -> None:
        """Register a simulation job; evict old finished jobs if needed."""
        self.sim_jobs[job.job_id] = job
        self._evict_from(self.sim_jobs)
        self.persist_job(job)
        emit_job_event("submitted", job, simulator=job.simulator)

    def add_batch_job(self, job: BatchJob) -> None:
        """Register a batch job; evict old finished batch jobs if needed."""
        self.batch_jobs[job.job_id] = job
        self._evict_from(self.batch_jobs)
        self.persist_job(job)
        emit_job_event("submitted", job, total_runs=job.total_runs)

    def _evict_from(self, jobs_dict: dict) -> None:
        """Evict oldest terminal jobs from a single dict when over the limit.

        When persistence is enabled, the on-disk record is deleted alongside
        the in-memory entry so the two never drift. Any per-job persistence
        lock is dropped here — safe once the job is out of the dict because
        no new ``persist_job`` calls can target it.
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_job(self, job: SimulationJob | BatchJob) -> None:
        """Write a job's current state to its per-circuit sidecar file.

        When called from an asyncio event loop, the file IO is scheduled on
        a worker thread so the loop doesn't stall on slow filesystems (WSL
        cross-filesystem, network mounts). Successive writes for the same
        ``job_id`` are serialised through ``_persist_locks`` so on-disk
        order matches call order — a "completed" write after "running"
        always wins.
        """
        if not self.persist_enabled:
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

    async def _persist_async(self, job: SimulationJob | BatchJob) -> None:
        """Serialise writes for a single job id; swallow and log failures."""
        lock = self._persist_locks.get(job.job_id)
        if lock is None:
            lock = self._persist_locks.setdefault(job.job_id, asyncio.Lock())
        async with lock:
            await asyncio.to_thread(self._persist_sync, job)

    def _persist_sync(self, job: SimulationJob | BatchJob) -> None:
        try:
            from ltspice_mcp.lib import job_store

            job_store.save_job(job)
        except Exception as e:
            # Persistence failures must never break simulation flow.
            logger.warning("Failed to persist job %s: %s", job.job_id, e)

    def persist_batch_progress(self, batch_job: BatchJob) -> None:
        """Persist a batch job's in-progress state, throttled by run count.

        Per-run callbacks for sweeps and Monte Carlo can fire thousands of
        times per job; serialising the full ``run_results`` dict on each
        call is O(N²). Write only on a sparse schedule so crash-recovery
        sees near-current state without paying the quadratic IO cost.
        """
        if not self.persist_enabled:
            return
        total = batch_job.total_runs
        done = batch_job.completed_runs
        # Checkpoint ~20 times per batch plus always on the final run.
        step = max(1, total // 20) if total else 1
        if done == total or done % step == 0:
            self.persist_job(batch_job)

    def _delete_persisted(self, job: SimulationJob | BatchJob) -> None:
        """Remove a job's on-disk record (used on eviction)."""
        if not self.persist_enabled:
            return
        try:
            from ltspice_mcp.lib import job_store

            job_store.delete_job(job)
        except Exception as e:
            logger.debug("Failed to delete persisted job %s: %s", job.job_id, e)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def ensure_loaded_for(self, circuit_path: Path) -> None:
        """Load any persisted jobs for this circuit into memory, once per session.

        No-op when persistence is disabled, the path is not a circuit file,
        or the sidecar directory doesn't exist. Jobs in non-terminal states
        at load time are marked ``interrupted`` (their owning server is gone).
        """
        if not self.persist_enabled:
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
            if sj.job_id in self.sim_jobs:
                continue
            self.sim_jobs[sj.job_id] = sj
            # If the sim outputs exist on disk, the job may have finished
            # just before the crash — promote interrupted → completed via
            # the recovery path so the emitted event is
            # 'interrupted_recovered', not 'completed'.
            if sj.status == "interrupted" and _has_valid_raw(sj.raw_file):
                sj.error = None
                # No state arg — the registry owns persistence below.
                recover(sj, "completed")
                self.persist_job(sj)
            elif sj.status == "interrupted":
                emit_job_event("interrupted_recovered", sj, recovered_as="interrupted")
        for bj in batch_jobs:
            if bj.job_id in self.batch_jobs:
                continue
            self.batch_jobs[bj.job_id] = bj
            if bj.status == "interrupted":
                emit_job_event("interrupted_recovered", bj, recovered_as="interrupted")

    def preload_recent(self, max_circuits: int = 10) -> int:
        """Eager-load persisted jobs for the ``max_circuits`` most recently
        touched circuits so first-tool-call latency doesn't spike.

        Returns the number of circuits actually loaded. No-op when
        persistence is disabled or ``max_circuits`` is 0. Circuits whose
        files have disappeared are pruned from the recent index as a
        side-effect. Failures for any one circuit are swallowed and logged
        — the lazy ``ensure_loaded_for`` path remains as a fallback.

        Subsequent ``ensure_loaded_for`` calls for the same paths are
        no-ops (deduped via ``_loaded_circuits``).
        """
        if not self.persist_enabled or max_circuits <= 0:
            return 0
        try:
            from ltspice_mcp.lib import recent

            entries = recent.load(prune_missing=True)[:max_circuits]
        except Exception as e:
            logger.debug("preload_recent: failed to read recent index: %s", e)
            return 0

        loaded = 0
        for entry in entries:
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            try:
                self.ensure_loaded_for(Path(raw_path))
                loaded += 1
            except Exception as e:
                logger.debug("preload_recent: skipped %s: %s", raw_path, e)
        logger.debug(
            "preload_recent: loaded %d circuit(s) from recent index", loaded
        )
        return loaded

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def drain_pending(self) -> None:
        """Wait for any outstanding persistence writes to complete."""
        if self._pending_persist:
            await asyncio.gather(*self._pending_persist, return_exceptions=True)

    async def cancel_running(self, runners, session_state) -> None:
        """Cancel any jobs still in running/queued state.

        Runners' ``cancel`` APIs take a SessionState for historical reasons;
        the caller passes it through rather than the registry reaching back
        for a circular reference.
        """
        sim_runner = runners.get_existing_sim_runner()
        for job in self.sim_jobs.values():
            if job.status in NON_TERMINAL_LIVE_STATUSES:
                if sim_runner is not None:
                    await sim_runner.cancel(job, session_state)
                else:
                    transition(job, "cancelled")
                    self.persist_job(job)

        sweep_runner = runners.get_existing_sweep_runner()
        mc_runner = runners.get_existing_mc_runner()
        for batch_job in self.batch_jobs.values():
            if batch_job.status == "running":
                if batch_job.job_type == "sweep" and sweep_runner is not None:
                    await sweep_runner.cancel(batch_job, session_state)
                elif batch_job.job_type == "montecarlo" and mc_runner is not None:
                    await mc_runner.cancel(batch_job, session_state)
                else:
                    transition(batch_job, "cancelled")
                    self.persist_job(batch_job)
            if batch_job.task is not None and not batch_job.task.done():
                batch_job.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await batch_job.task
