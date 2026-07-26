"""In-memory registry for simulation, batch, and experiment jobs.

Owns the single union ``jobs`` dict
plus all disk-persistence coordination (sidecar writes, eviction,
interrupted-job recovery). Split out of ``SessionState`` so the
per-session container stays focused on simulator catalog, caches, and
configuration.

``SessionState`` delegates its job-facing API to this class; call sites
continue to use ``state.jobs``, ``state.add_job``, etc. The ``sim_jobs``,
``batch_jobs``, and ``experiment_jobs`` attributes are type-filtered
writable views over the union store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Iterator, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import TERMINAL_CASE_STATUSES, ExperimentJob
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
Job = SimulationJob | BatchJob | ExperimentJob
J = TypeVar("J", bound=Job)

# Maximum finished jobs to retain per job type (single-sim, batch, experiment).
_MAX_FINISHED_JOBS = 200

# How long shutdown waits on one stage of cancels (see ``_issue_cancels``).
# Every such wait is on live work that may never end — a simulator process
# wedged past its kill, a task that swallows its cancellation — so an unbounded
# await would hold shutdown open indefinitely, and with it the job-persistence
# flush that follows.
_SHUTDOWN_CANCEL_TIMEOUT_S = 10.0

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


def _discard_outcome(task: asyncio.Future[Any]) -> None:
    """Retrieve a finished cancel's result so asyncio does not log it unhandled."""
    with contextlib.suppress(BaseException):
        task.exception()


async def _issue_cancels(cancels: list[Awaitable[Any]]) -> None:
    """Run one shutdown stage's cancels: together, bounded, and isolated.

    Every cancel here waits on live work — a runner's kill of a simulator
    process, or a job's own task winding down — so any of them can hang or
    raise outright (a runner refuses when it no longer owns the job).
    Awaited one by one, the first such failure ends the whole shutdown: the
    stages after it are never reconciled, and neither is the persistence flush
    that would have recorded them. So every outcome is discarded rather than
    raised; what a failed cancel leaves behind is a job still in a live status,
    which the caller's own bookkeeping then finishes.

    Issued together so the timeout bounds the stage rather than each job: a
    per-job wait would multiply a client's shutdown grace by the number of live
    jobs, spending on cancels the time the flush needs.

    ``asyncio.wait`` and not ``wait_for``, because ``wait_for`` bounds only the
    cooperative case: on timeout it cancels its awaitable and then *awaits that
    cancellation*, so anything that declines to stop — the very case this bound
    exists for — hangs it as surely as a bare await. ``wait`` returns on the
    deadline regardless; the stragglers are asked to stop and left to it,
    because this process is exiting anyway and the flush is waiting.
    """
    if not cancels:
        return
    tasks = [asyncio.ensure_future(cancel) for cancel in cancels]
    for task in tasks:
        task.add_done_callback(_discard_outcome)
    _, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_CANCEL_TIMEOUT_S)
    for task in pending:
        task.cancel()


def _cancel_tasks(jobs: list[BatchJob] | list[ExperimentJob]) -> list[Awaitable[Any]]:
    """Cancel each job's still-live task; return the awaits for the bound above.

    Requesting cancellation is not the same as being stopped: a task that
    swallows ``CancelledError``, or is blocked inside a shielded section, keeps
    its await open for as long as it likes. Unbounded, that stalls shutdown
    exactly as a wedged runner cancel does — and the persistence flush is still
    behind it.
    """
    pending: list[Awaitable[Any]] = []
    for job in jobs:
        task = job.task
        if task is not None and not task.done():
            task.cancel()
            pending.append(task)
    return pending


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

    Rule for new code: use the typed view (``registry.sim_jobs``,
    ``registry.batch_jobs``, or ``registry.experiment_jobs``) when the code
    is scoped to one job type; use
    ``registry.jobs`` / ``state.all_jobs`` plus ``isinstance`` when handling
    either type.
    """

    def __init__(self, store: dict[str, Job], job_type: type[J]) -> None:
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
    """Tracks all job kinds with optional disk persistence.

    Attributes:
        persist_enabled: When True, sidecar files are written alongside
            circuits and evictions delete them. When False, the registry
            behaves as a pure in-memory store.
        jobs: The single source of truth for every job regardless of run
            type. ``sim_jobs``, ``batch_jobs``, and ``experiment_jobs`` are
            type-filtered views over it.
    """

    persist_enabled: bool
    working_dir: Path = field(default_factory=Path.cwd)
    jobs: dict[str, Job] = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)
    _loaded_circuits: set[Path] = field(default_factory=set, repr=False)
    """Resolved circuit paths whose persisted jobs have been loaded this session."""
    _pending_persist: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    """In-flight persistence writes; drained on shutdown."""
    _persist_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    """Per-job-id locks serialising successive writes.

    Eviction removes a lock only after earlier writes and the persisted-record
    deletion have completed. Removing it inside a write would let a new writer
    allocate a second lock while the old one is still held.
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

    @property
    def experiment_jobs(self) -> _TypedJobView[ExperimentJob]:
        """Writable view of multi-circuit experiment jobs."""
        return _TypedJobView(self.jobs, ExperimentJob)

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

    def add_experiment_job(
        self,
        job: ExperimentJob,
        *,
        already_persisted: bool = False,
    ) -> None:
        """Register an experiment after its durable receipt barrier."""
        self.jobs[job.job_id] = job
        self._evict_from(self.experiment_jobs)
        if not already_persisted:
            self.persist_job(job)
        emit_job_event("submitted", job, total_cases=job.completeness.expanded)

    def _evict_from(self, jobs_view: MutableMapping[str, J]) -> None:
        """Evict oldest terminal jobs of one job type when over the limit.

        ``jobs_view`` is a typed view over the union store, so the cap is
        enforced per job type (200 finished jobs of each kind). When
        persistence is enabled, the on-disk record is
        deleted alongside the in-memory entry so the two never drift. Async
        deletion drains earlier writes before it drops the per-job lock.
        """
        finished = [(jid, j) for jid, j in jobs_view.items() if j.status in TERMINAL_STATUSES]
        overflow = len(finished) - _MAX_FINISHED_JOBS
        if overflow <= 0:
            return
        finished.sort(key=lambda pair: pair[1].started_at)
        for jid, j in finished[:overflow]:
            del jobs_view[jid]
            self._delete_persisted(j)

    def _load_foreign_job_sync(self, job: Job) -> Job | None:
        """Blocking store dispatch for refreshing one foreign-owned job."""
        if isinstance(job, ExperimentJob):
            from ltspice_mcp.lib import experiment_store

            return experiment_store.load_job_from_path(job.store_path, self.working_dir)

        from ltspice_mcp.lib import job_store

        return job_store.load_job(job.job_id, job.netlist)

    def refresh_foreign_job(self, job: Job) -> Job:
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
            fresh = self._load_foreign_job_sync(job)
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
        if isinstance(fresh, ExperimentJob) and any(
            item.get("code") == "server_restarted" for item in fresh.observations
        ):
            self.persist_job(fresh)
        return fresh

    async def refresh_foreign_job_async(self, job: Job) -> Job:
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
            fresh = await asyncio.to_thread(self._load_foreign_job_sync, job)
        except Exception as e:
            logger.debug("refresh_foreign_job %s: %s", job.job_id, e)
            return job
        if fresh is None:
            return job
        # On the loop here (awaited from a handler) — safe to swap the entry.
        self.jobs[job.job_id] = fresh
        if isinstance(fresh, ExperimentJob) and any(
            item.get("code") == "server_restarted" for item in fresh.observations
        ):
            self.persist_job(fresh)
        return fresh

    def refreshed_jobs(self) -> list[Job]:
        """Snapshot of every job, with parallel sessions' live jobs re-read.

        The listing surfaces (``check_job`` with no id, the results resource)
        call this so a foreign job doesn't show "running" forever after its
        owner finished it. Own and terminal jobs pass through unchanged.
        """
        return [self.refresh_foreign_job(job) for job in list(self.jobs.values())]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_job(self, job: Job) -> None:
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

    async def _persist_async(self, job: Job) -> None:
        """Serialise writes for a single job id; swallow and log failures."""
        lock = self._persist_locks.get(job.job_id)
        if lock is None:
            lock = self._persist_locks.setdefault(job.job_id, asyncio.Lock())
        async with lock:
            await asyncio.to_thread(self._persist_sync, job)

    def _persist_sync(self, job: Job) -> None:
        try:
            if isinstance(job, ExperimentJob):
                from ltspice_mcp.lib import experiment_store

                experiment_store.save_job(job)
            else:
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

    def _delete_persisted(self, job: Job) -> None:
        """Remove a job's on-disk record (used on eviction)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._delete_persisted_sync(job)
            self._persist_locks.pop(job.job_id, None)
            return
        task = loop.create_task(self._delete_persisted_async(job))
        self._pending_persist.add(task)
        task.add_done_callback(self._pending_persist.discard)

    async def _delete_persisted_async(self, job: Job) -> None:
        """Delete only after earlier writes for the same job have drained."""
        lock = self._persist_locks.get(job.job_id)
        if lock is None:
            lock = self._persist_locks.setdefault(job.job_id, asyncio.Lock())
        try:
            async with lock:
                await asyncio.to_thread(self._delete_persisted_sync, job)
        finally:
            self._persist_locks.pop(job.job_id, None)

    def _delete_persisted_sync(self, job: Job) -> None:
        """Blocking deletion half, including dependent immutable result sets."""
        try:
            if self.persist_enabled:
                if isinstance(job, ExperimentJob):
                    from ltspice_mcp.lib import experiment_store

                    experiment_store.delete_job(job, self.working_dir)
                else:
                    from ltspice_mcp.lib import job_store

                    job_store.delete_job(job)
            from ltspice_mcp.lib import result_store

            result_store.invalidate_for_job(self.working_dir, job.job_id)
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

    def _read_persisted_jobs(
        self,
        resolved: Path,
    ) -> tuple[list, list, list, list] | None:
        """File-read half of the load — offloadable (touches no registry state)."""
        try:
            from ltspice_mcp.lib import experiment_store, job_store

            sim_jobs, batch_jobs = job_store.load_jobs_for_circuit(resolved)
            experiment_jobs, observations = experiment_store.load_pointer_jobs(
                resolved,
                self.working_dir,
            )
            return sim_jobs, batch_jobs, experiment_jobs, observations
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
        applied = False
        try:
            loaded = await asyncio.to_thread(self._read_persisted_jobs, resolved)
            if loaded is not None:
                self._apply_loaded_jobs(*loaded)
            applied = True
        finally:
            # The claim goes in before the cancellable read; if that read is
            # cancelled (or the apply raises), release it so a later call
            # retries instead of deduping to a permanent no-op that would leave
            # an on-disk job unloadable. A failed read (loaded is None) keeps
            # the claim, matching the sync path.
            if not applied:
                self._loaded_circuits.discard(resolved)

    def _apply_loaded_jobs(
        self,
        sim_jobs: list[SimulationJob],
        batch_jobs: list[BatchJob],
        experiment_jobs: list[ExperimentJob],
        observations: list[dict],
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
        for experiment in experiment_jobs:
            if experiment.job_id in self.jobs:
                continue
            self.jobs[experiment.job_id] = experiment
            restarted = any(
                item.get("code") == "server_restarted" for item in experiment.observations
            )
            if experiment.status == "interrupted" or restarted:
                emit_job_event(
                    "interrupted_recovered",
                    experiment,
                    recovered_as=experiment.status,
                )
            if restarted:
                self.persist_job(experiment)
        self.observations.extend(observations)

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
        sim_cancels: list[Awaitable[Any]] = []
        for job in list(self.sim_jobs.values()):
            if job.status in NON_TERMINAL_LIVE_STATUSES and job.owner_pid == own_pid:
                # Match the runner to the job's own simulator: runners are
                # cached per simulator class, and the kill scopes by that
                # class's executable names.
                sim_runner = runners.get_existing_sim_runner(job.simulator)
                if sim_runner is not None:
                    sim_cancels.append(sim_runner.cancel(job, session_state))
                else:
                    transition(job, "cancelled")
                    self.persist_job(job)
        await _issue_cancels(sim_cancels)

        batch_cancels: list[Awaitable[Any]] = []
        for batch_job in list(self.batch_jobs.values()):
            if batch_job.status == "running" and batch_job.owner_pid == own_pid:
                # Route to the runner instance that launched the batch — with
                # several runners of one kind cached, most-recent isn't
                # necessarily the owner of this job's cancel event.
                batch_runner = runners.get_batch_runner_for(batch_job)
                if batch_runner is not None:
                    batch_cancels.append(batch_runner.cancel(batch_job, session_state))
                else:
                    transition(batch_job, "cancelled")
                    self.persist_job(batch_job)
        await _issue_cancels(batch_cancels)

        await _issue_cancels(_cancel_tasks(list(self.batch_jobs.values())))

        experiments = list(self.experiment_jobs.values())
        await _issue_cancels(
            [
                runner.cancel(experiment)
                for experiment in experiments
                if experiment.owner_pid == own_pid
                and (runner := runners.get_experiment_runner_for(experiment)) is not None
            ]
        )
        # No record is kept of which cancels succeeded, because a cancel that
        # RETURNED already left its job terminal — every return path of the
        # coordinator's cancel is behind the job's done event, and only a
        # terminal transition sets that. So the status guard below is the whole
        # test: a job still non-terminal here is one whose cancel timed out,
        # raised, or never existed, and this pass is the last thing that can
        # write it a terminal status before the process exits. Skipping it would
        # persist a sidecar reading "running" under a pid that no longer exists.
        for experiment in experiments:
            if experiment.status in NON_TERMINAL_LIVE_STATUSES and experiment.owner_pid == own_pid:
                cancelled_at = now()
                newly_cancelled = [
                    case for case in experiment.cases if case.status not in TERMINAL_CASE_STATUSES
                ]
                experiment.cases = [
                    (
                        case
                        if case.status in TERMINAL_CASE_STATUSES
                        else replace(
                            case,
                            status="cancelled",
                            failure_code="server_shutdown",
                            error="Server shut down before this case completed",
                            completed_at=cancelled_at,
                        )
                    )
                    for case in experiment.cases
                ]
                experiment.completeness = replace(
                    experiment.completeness,
                    cancelled=(experiment.completeness.cancelled + len(newly_cancelled)),
                )
                experiment.failures.extend(
                    {
                        "case_id": case.case_id,
                        "code": "server_shutdown",
                        "message": "Server shut down before this case completed",
                    }
                    for case in newly_cancelled
                )
                if experiment.analysis.status in {"pending", "running"}:
                    experiment.analysis = replace(
                        experiment.analysis,
                        status="cancelled",
                        error="Server shut down before attached analysis completed",
                        completed_at=cancelled_at,
                    )
                experiment.runs_done_event.set()
                transition(experiment, "cancelled")
                self.persist_job(experiment)

        await _issue_cancels(_cancel_tasks(experiments))
