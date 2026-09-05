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
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import TERMINAL_CASE_STATUSES, ExperimentJob
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    LegacyJobRecord,
)
from ltspice_mcp.lib.observability import emit_job_event

logger = logging.getLogger(__name__)

# Bound to the union job type so the typed views and per-type eviction stay
# scoped to one job class at a time.
Job = LegacyJobRecord | ExperimentJob
J = TypeVar("J", bound=Job)

# Maximum finished jobs to retain per job type (single-sim, batch, experiment).
_MAX_FINISHED_JOBS = 200

# How long shutdown waits on one stage of cancels (see ``_issue_cancels``).
# Every such wait is on live work that may never end — a simulator process
# wedged past its kill, a task that swallows its cancellation — so an unbounded
# await would hold shutdown open indefinitely, and with it the job-persistence
# flush that follows.
_SHUTDOWN_CANCEL_TIMEOUT_S = 10.0


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


def _cancel_tasks(jobs: list[ExperimentJob]) -> list[Awaitable[Any]]:
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
    def legacy_records(self) -> _TypedJobView[LegacyJobRecord]:
        """Writable view of the pre-0.6 records loaded from disk."""
        return _TypedJobView(self.jobs, LegacyJobRecord)

    @property
    def experiment_jobs(self) -> _TypedJobView[ExperimentJob]:
        """Writable view of multi-circuit experiment jobs."""
        return _TypedJobView(self.jobs, ExperimentJob)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

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
        finished.sort(key=lambda pair: getattr(pair[1], "started_at", None) or 0)
        for jid, j in finished[:overflow]:
            del jobs_view[jid]
            self._delete_persisted(j)

    # ------------------------------------------------------------------
    # Lookup — the one route from a job id to a job
    # ------------------------------------------------------------------

    @staticmethod
    def _on_event_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _adopt(self, job: Job) -> Job:
        """Take a freshly-read record into the registry, if this caller may.

        Registry mutations are loop-only, the same contract that governs the
        cached editors (see ``tools/_base.py``): a worker thread swapping an
        entry could race a loop-side transition on the same job. So an off-loop
        caller — a resource read, a preload — gets the record it just read as a
        read-only view and the registry is left untouched; the next on-loop
        resolution adopts it. This is the ONLY place that decides between the
        two, so no caller can quietly pick the other answer.

        A record the store had to reconcile (its owner died mid-run) is written
        back, because the reconciliation is a fact about the job that nothing
        else will persist.
        """
        if not self._on_event_loop():
            return job
        self.jobs[job.job_id] = job
        if isinstance(job, ExperimentJob) and any(
            item.get("code") == "server_restarted" for item in job.observations
        ):
            self.persist_job(job)
        return job

    def _load_from_store(self, job_id: str) -> ExperimentJob | None:
        """Blocking read of one experiment record from this session's store."""
        if not self.persist_enabled:
            return None
        from ltspice_mcp.lib import experiment_store

        return experiment_store.load_job(job_id, self.working_dir, own_is_alive=True)

    def get_or_load(self, job_id: str) -> Job | None:
        """A job by id: in memory, else from the store. None if there is none.

        The single discovery route. Everything that resolves an id — the tools,
        the resources, the Python API — comes through here, so "the registry
        did not have it" and "the store did not have it either" are one answer
        rather than a sequence of fallbacks each caller re-assembles.

        Raises ``ValueError`` for an id that could never name a record.
        """
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        from ltspice_mcp.lib.store import validate_job_id

        validate_job_id(job_id)
        loaded = self._load_from_store(job_id)
        return self._adopt(loaded) if loaded is not None else None

    async def get_or_load_async(self, job_id: str) -> Job | None:
        """Loop-safe ``get_or_load``: offload the store read, adopt on the loop."""
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        from ltspice_mcp.lib.store import validate_job_id

        validate_job_id(job_id)
        loaded = await asyncio.to_thread(self._load_from_store, job_id)
        return self._adopt(loaded) if loaded is not None else None

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
            or getattr(job, "owner_pid", 0) in (0, os.getpid())
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
        return self._adopt(fresh)

    async def refresh_foreign_job_async(self, job: Job) -> Job:
        """Loop-safe ``refresh_foreign_job``: offload the sidecar re-read.

        Same contract as the sync version, but the single-file ``load_job``
        read runs in a worker thread so a wedged filesystem can't freeze the
        loop. The guard (no IO for own/terminal jobs) and the registry swap
        both stay on the loop; only the foreign non-terminal case does IO.
        """
        if (
            not self.persist_enabled
            or getattr(job, "owner_pid", 0) in (0, os.getpid())
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
        # On the loop here (awaited from a handler), so ``_adopt`` swaps it in.
        return self._adopt(fresh)

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
            await self._offload_persistence(self._persist_sync, job)

    async def _offload_persistence(self, fn: Callable[[Job], None], job: Job) -> None:
        """Run one blocking persistence step off-loop, surviving teardown.

        During interpreter teardown the default executor is gone and
        ``asyncio.to_thread`` raises "cannot schedule new futures after
        shutdown" — as an unretrieved task exception it printed a scary
        irrelevant traceback while the write it carried was silently lost
        (observed live: a ``wait=False`` script exiting while its job
        settled). Blocking is fine during teardown; run synchronously.
        Shared by the write and delete halves — a lost delete resurrects a
        stale sidecar as a job on the next preload.
        """
        try:
            await asyncio.to_thread(fn, job)
        except RuntimeError:
            fn(job)

    def _persist_sync(self, job: Job) -> None:
        # A legacy record is read-only: this version never wrote it and has
        # nothing new to say about it, so persisting one would only risk
        # rewriting an earlier release's file in a shape it cannot read.
        if not isinstance(job, ExperimentJob):
            return
        try:
            from ltspice_mcp.lib import experiment_store

            experiment_store.save_job(job)
        except Exception as e:
            # Persistence failures must never break simulation flow.
            logger.warning("Failed to persist job %s: %s", job.job_id, e)

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
                await self._offload_persistence(self._delete_persisted_sync, job)
        finally:
            self._persist_locks.pop(job.job_id, None)

    def _delete_persisted_sync(self, job: Job) -> None:
        """Blocking deletion half, including dependent immutable result sets."""
        try:
            if self.persist_enabled and isinstance(job, ExperimentJob):
                from ltspice_mcp.lib import experiment_store

                experiment_store.delete_job(job, self.working_dir)
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
    ) -> tuple[list, list, list] | None:
        """File-read half of the load — offloadable (touches no registry state)."""
        try:
            from ltspice_mcp.lib import experiment_store, job_store

            legacy_records = job_store.load_jobs_for_circuit(resolved)
            experiment_jobs, observations = experiment_store.load_jobs_for_circuit(
                resolved,
                self.working_dir,
            )
            return legacy_records, experiment_jobs, observations
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
        legacy_records: list[LegacyJobRecord],
        experiment_jobs: list[ExperimentJob],
        observations: list[dict],
    ) -> None:
        """Registry-mutation half of the load — loop-only (mutates ``self.jobs``).

        A legacy record is registered so a caller asking about it is told what
        it is; there is no recovery to attempt, because this version has no
        runner that could resume or re-read it.
        """
        for record in legacy_records:
            if record.job_id not in self.jobs:
                self.jobs[record.job_id] = record
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
        # Snapshot the view before iterating: the typed view iterates the live
        # union dict lazily, and the awaits below suspend this coroutine — a
        # concurrent job registration during a cancel would otherwise raise
        # "dictionary changed size during iteration".
        #
        # Only THIS process's jobs are cancelled: a parallel server session's
        # live job also sits in the registry as running (loaded from its
        # sidecar with the owner still alive) and must not be killed or
        # relabeled by our shutdown. A legacy record is never running under
        # this version — nothing here could have launched one.
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
