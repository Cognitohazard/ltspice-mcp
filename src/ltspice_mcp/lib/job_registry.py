"""In-memory registry for simulation and batch jobs.

Owns the single union ``jobs`` dict (job_id -> SimulationJob | BatchJob)
plus all disk-persistence coordination (sidecar writes, eviction,
interrupted-job recovery). Split out of ``SessionState`` so the
per-session container stays focused on simulator catalog, caches, and
configuration.

``SessionState`` delegates its job-facing API to this class; call sites
continue to use ``state.jobs``, ``state.add_job``, etc. The ``sim_jobs``
and ``batch_jobs`` attributes are type-filtered writable views over the
union store, so per-type call sites keep their old dict semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from ltspice_mcp.lib.job_lifecycle import recover, transition
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    SimulationJob,
)
from ltspice_mcp.lib.observability import emit_job_event

logger = logging.getLogger(__name__)

# Bound to the union job type so the typed views and per-type eviction stay
# scoped to one job class at a time.
J = TypeVar("J", bound=SimulationJob | BatchJob)

# Maximum finished jobs to retain per job type (single-sim, batch).
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


class _TypedJobView(MutableMapping[str, J]):
    """Permanent typed access layer over the union job store.

    This is the type-scoped surface of the registry: per-type eviction caps,
    type-scoped iteration for resources and status reporting, and
    write-through with a runtime type guard. Lookups (``[]``, ``get``,
    ``in``), iteration, and ``len`` surface only entries of the view's job
    type — a batch id accessed through the sim view behaves as absent, and
    vice versa. Writes (``view[key] = job``) go straight through to the
    union dict but reject values of the wrong job type, and ``del`` removes
    only entries of the view's type.

    Rule for new code: use the typed view (``registry.sim_jobs`` /
    ``registry.batch_jobs``) when the code is scoped to one job type; use
    ``registry.jobs`` / ``state.all_jobs`` plus ``isinstance`` when handling
    either type.
    """

    def __init__(self, store: dict[str, SimulationJob | BatchJob], job_type: type[J]) -> None:
        self._store = store
        self._job_type = job_type

    def __getitem__(self, key: str) -> J:
        job = self._store[key]
        if not isinstance(job, self._job_type):
            raise KeyError(key)
        return job

    def __setitem__(self, key: str, value: J) -> None:
        # Guard at runtime: a wrong-type job written through this view would
        # land in the union store but be invisible through the view that
        # stored it — a silent misroute that static typing alone can't stop.
        # Widen to ``object`` so the type checker keeps the failure branch
        # live: with the parameter typed ``J`` it narrows the negative
        # isinstance branch to Never, but untyped callers reach it at runtime.
        candidate: object = value
        if not isinstance(candidate, self._job_type):
            raise TypeError(
                f"{self._job_type.__name__} view cannot store {type(value).__name__} (key {key!r})"
            )
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        if not isinstance(self._store[key], self._job_type):
            raise KeyError(key)
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return (k for k, v in self._store.items() if isinstance(v, self._job_type))

    def __len__(self) -> int:
        return sum(1 for v in self._store.values() if isinstance(v, self._job_type))


@dataclass
class JobRegistry:
    """Tracks simulation and batch jobs with optional disk persistence.

    Attributes:
        persist_enabled: When True, sidecar files are written alongside
            circuits and evictions delete them. When False, the registry
            behaves as a pure in-memory store.
        jobs: job_id -> SimulationJob | BatchJob — the single source of
            truth for every job regardless of run type. ``sim_jobs`` /
            ``batch_jobs`` are type-filtered views over it.
    """

    persist_enabled: bool
    jobs: dict[str, SimulationJob | BatchJob] = field(default_factory=dict)
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
    # Typed views
    # ------------------------------------------------------------------

    @property
    def sim_jobs(self) -> _TypedJobView[SimulationJob]:
        """Writable view of the single-simulation jobs in the union store."""
        return _TypedJobView(self.jobs, SimulationJob)

    @property
    def batch_jobs(self) -> _TypedJobView[BatchJob]:
        """Writable view of the batch (sweep/MC) jobs in the union store."""
        return _TypedJobView(self.jobs, BatchJob)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_sim_job(self, job: SimulationJob) -> None:
        """Register a simulation job; evict old finished jobs if needed."""
        self.jobs[job.job_id] = job
        self._evict_from(self.sim_jobs)
        self.persist_job(job)
        emit_job_event("submitted", job, simulator=job.simulator)

    def add_batch_job(self, job: BatchJob) -> None:
        """Register a batch job; evict old finished batch jobs if needed."""
        self.jobs[job.job_id] = job
        self._evict_from(self.batch_jobs)
        self.persist_job(job)
        emit_job_event("submitted", job, total_runs=job.total_runs)

    def _evict_from(self, jobs_view: MutableMapping[str, J]) -> None:
        """Evict oldest terminal jobs of one job type when over the limit.

        ``jobs_view`` is a typed view over the union store, so the cap is
        enforced per job type (200 finished single-sim jobs AND 200 finished
        batch jobs). When persistence is enabled, the on-disk record is
        deleted alongside the in-memory entry so the two never drift. Any
        per-job persistence lock is dropped here — safe once the job is out
        of the store because no new ``persist_job`` calls can target it.
        """
        finished = [(jid, j) for jid, j in jobs_view.items() if j.status in TERMINAL_STATUSES]
        overflow = len(finished) - _MAX_FINISHED_JOBS
        if overflow <= 0:
            return
        finished.sort(key=lambda pair: pair[1].started_at)
        for jid, j in finished[:overflow]:
            del jobs_view[jid]
            self._delete_persisted(j)
            self._persist_locks.pop(jid, None)

    def refresh_foreign_job(self, job: SimulationJob | BatchJob) -> SimulationJob | BatchJob:
        """Re-read a parallel session's live job from its sidecar.

        A job loaded while its owning process was alive sits in this
        registry as running/queued, but only the owner updates it — nothing
        in this process would ever see it finish. Re-reading the sidecar at
        resolution time picks up the owner's latest persisted state
        (including the interrupted translation once the owner has died).
        Own jobs, terminal jobs, and persistence-off sessions return
        unchanged.
        """
        if (
            not self.persist_enabled
            or job.owner_pid in (0, os.getpid())
            or job.status not in NON_TERMINAL_LIVE_STATUSES
        ):
            return job
        try:
            from ltspice_mcp.lib import job_store

            fresh = job_store.load_job(job.job_id, job.netlist)
        except Exception as e:
            logger.debug("refresh_foreign_job %s: %s", job.job_id, e)
            return job
        if fresh is None:
            return job
        # Registry mutations are loop-only (the same contract as the cached
        # editors — see tools/_base.py): resource reads run this via a worker
        # thread (server.py offloads whole resource reads), where swapping the
        # entry could race a loop-side transition on the same job. Off-loop
        # callers get the fresh view without the registry update; the next
        # on-loop resolution persists it.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return fresh
        self.jobs[job.job_id] = fresh
        return fresh

    async def refresh_foreign_job_async(
        self, job: SimulationJob | BatchJob
    ) -> SimulationJob | BatchJob:
        """Loop-safe ``refresh_foreign_job``: offload the sidecar re-read.

        Same contract as the sync version, but the single-file ``load_job``
        read runs in a worker thread so a wedged filesystem can't freeze the
        loop. The guard (no IO for own/terminal jobs) and the registry swap
        both stay on the loop; only the foreign non-terminal case does IO.
        """
        if (
            not self.persist_enabled
            or job.owner_pid in (0, os.getpid())
            or job.status not in NON_TERMINAL_LIVE_STATUSES
        ):
            return job
        try:
            from ltspice_mcp.lib import job_store

            fresh = await asyncio.to_thread(job_store.load_job, job.job_id, job.netlist)
        except Exception as e:
            logger.debug("refresh_foreign_job %s: %s", job.job_id, e)
            return job
        if fresh is None:
            return job
        # On the loop here (awaited from a handler) — safe to swap the entry.
        self.jobs[job.job_id] = fresh
        return fresh

    def refreshed_jobs(self) -> list[SimulationJob | BatchJob]:
        """Snapshot of every job, with parallel sessions' live jobs re-read.

        The listing surfaces (``check_job`` with no id, the results resource)
        call this so a foreign job doesn't show "running" forever after its
        owner finished it. Own and terminal jobs pass through unchanged.
        """
        return [self.refresh_foreign_job(job) for job in list(self.jobs.values())]

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

    def _claim_circuit_load(self, circuit_path: Path) -> Path | None:
        """Resolve + dedup a circuit for a one-shot load; None if nothing to do.

        Adds to ``_loaded_circuits`` BEFORE the read so a concurrent dispatch
        for the same circuit skips it (benign: the second caller proceeds
        without the jobs for the brief read window, then sees them applied).
        """
        if not self.persist_enabled:
            return None
        try:
            resolved = circuit_path.resolve()
        except OSError:
            return None
        if resolved in self._loaded_circuits:
            return None
        self._loaded_circuits.add(resolved)
        return resolved

    @staticmethod
    def _read_persisted_jobs(resolved: Path) -> tuple[list, list] | None:
        """File-read half of the load — offloadable (touches no registry state)."""
        try:
            from ltspice_mcp.lib import job_store

            return job_store.load_jobs_for_circuit(resolved)
        except Exception as e:
            logger.warning("Failed to load persisted jobs for %s: %s", resolved, e)
            return None

    def ensure_loaded_for(self, circuit_path: Path) -> None:
        """Load any persisted jobs for this circuit into memory, once per session.

        No-op when persistence is disabled, the path is not a circuit file,
        or the sidecar directory doesn't exist. Jobs in non-terminal states
        at load time are marked ``interrupted`` (their owning server is gone).

        Synchronous — for off-loop callers (startup ``preload_recent``). On the
        event loop use ``ensure_loaded_for_async`` so the sidecar read (a glob +
        JSON reads that stalls the whole loop on a wedged ``/mnt/c``) is offloaded.
        """
        resolved = self._claim_circuit_load(circuit_path)
        if resolved is None:
            return
        loaded = self._read_persisted_jobs(resolved)
        if loaded is not None:
            self._apply_loaded_jobs(*loaded)

    async def ensure_loaded_for_async(self, circuit_path: Path) -> None:
        """Loop-safe ``ensure_loaded_for``: offload the read, apply on the loop.

        The sidecar read runs in a worker thread (an unresponsive filesystem
        must not freeze the shared event loop — this runs on the common tool-
        dispatch path). The registry mutation stays on the loop, per the
        loop-only contract that also governs the cached editors.
        """
        resolved = self._claim_circuit_load(circuit_path)
        if resolved is None:
            return
        loaded = await asyncio.to_thread(self._read_persisted_jobs, resolved)
        if loaded is not None:
            self._apply_loaded_jobs(*loaded)

    def _apply_loaded_jobs(
        self,
        sim_jobs: list[SimulationJob],
        batch_jobs: list[BatchJob],
    ) -> None:
        """Registry-mutation half of the load — loop-only (mutates ``self.jobs``)."""
        for sj in sim_jobs:
            if sj.job_id in self.jobs:
                continue
            self.jobs[sj.job_id] = sj
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
            if bj.job_id in self.jobs:
                continue
            self.jobs[bj.job_id] = bj
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
        logger.debug("preload_recent: loaded %d circuit(s) from recent index", loaded)
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
        own_pid = os.getpid()
        # Snapshot both views before iterating: the typed views iterate the
        # live union dict lazily, and the awaits below suspend this coroutine
        # — a concurrent job registration during a cancel would otherwise
        # raise "dictionary changed size during iteration".
        #
        # Only THIS process's jobs are cancelled: a parallel server session's
        # live job also sits in the registry as running (loaded from its
        # sidecar with the owner still alive) and must not be killed or
        # relabeled by our shutdown.
        for job in list(self.sim_jobs.values()):
            if job.status in NON_TERMINAL_LIVE_STATUSES and job.owner_pid == own_pid:
                # Match the runner to the job's own simulator: runners are
                # cached per simulator class, and the kill scopes by that
                # class's executable names.
                sim_runner = runners.get_existing_sim_runner(job.simulator)
                if sim_runner is not None:
                    await sim_runner.cancel(job, session_state)
                else:
                    transition(job, "cancelled")
                    self.persist_job(job)

        for batch_job in list(self.batch_jobs.values()):
            if batch_job.status == "running" and batch_job.owner_pid == own_pid:
                # Route to the runner instance that launched the batch — with
                # several runners of one kind cached, most-recent isn't
                # necessarily the owner of this job's cancel event.
                batch_runner = runners.get_batch_runner_for(batch_job)
                if batch_runner is not None:
                    await batch_runner.cancel(batch_job, session_state)
                else:
                    transition(batch_job, "cancelled")
                    self.persist_job(batch_job)
            if batch_job.task is not None and not batch_job.task.done():
                batch_job.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await batch_job.task
