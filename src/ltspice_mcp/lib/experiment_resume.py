"""Admission and authority for linked, frozen-input experiment attempts."""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ltspice_mcp.errors import JobNotFoundError
from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.controlled_ngspice import (
    prepare_seeded_driver,
    simulator_command,
    verify_execution_policy,
    verify_seeded_driver,
)
from ltspice_mcp.lib.experiment_inputs import (
    capture_case_inputs,
    prepare_case_startup,
    prepare_startup,
    verify_case_inputs,
    verify_produced_artifacts,
    verify_startup,
)
from ltspice_mcp.lib.experiment_types import (
    AnalysisStage,
    ExperimentCase,
    ExperimentJob,
    failure_row,
)
from ltspice_mcp.lib.filelock import async_file_lock, file_lock
from ltspice_mcp.lib.native_execution import prepare_native_cases, prepare_native_retry
from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY
from ltspice_mcp.lib.proc_kill import (
    ProcessPresence,
    process_identity_presence,
    process_start_marker,
    simulator_executable_names,
    simulator_presence,
)
from ltspice_mcp.lib.recovery_journal import (
    JournalEntry,
    RecoveryJournal,
    add_child,
    add_noop,
    candidate_job,
    load_journal,
    mark_launched,
    new_root_journal,
    prepared_entry,
    save_journal,
)
from ltspice_mcp.lib.recovery_records import (
    CaseAttempt,
    ExecutionRecord,
    JobRecovery,
    ProcessIdentity,
    RecoveryError,
)
from ltspice_mcp.lib.simulator_build import executable_identity, same_executable
from ltspice_mcp.lib.store import OwnerLiveness, Store, path_digest
from ltspice_mcp.lib.sweep_utils import generate_id

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_runner import (
        AdmissionResult,
        AnalysisCallback,
        ExperimentReceipt,
        ExperimentRunner,
        ExperimentRunRequest,
        SubmissionCommitted,
    )
    from ltspice_mcp.state import SessionState


@dataclass(frozen=True)
class ResumeReceipt:
    job: ExperimentJob
    replayed: bool
    control_token: str | None
    resumed: bool


def _committed_failure(
    job: ExperimentJob, cause: Exception, *, replayed: bool = False
) -> SubmissionCommitted:
    from ltspice_mcp.lib.experiment_runner import ExperimentReceipt, SubmissionCommitted

    return SubmissionCommitted(
        job.request_id,
        cause,
        receipt=ExperimentReceipt(job, replayed, job.control_token),
    )


def refuse_reserved_request(request: ExperimentRunRequest) -> None:
    """An ordinary submission cannot bypass a journal-only committed root."""
    store = Store(request.state.working_dir)
    if load_journal(store, request.request_id) is None:
        return
    from ltspice_mcp.lib.experiment_runner import IdempotencyConflictError

    raise IdempotencyConflictError("request_id already belongs to a recoverable experiment")


def authorize_resume(job: ExperimentJob, control_token: str | None) -> str:
    """Return authority only to the actual owner or a holder of its capability."""
    recovery = job.recovery
    if recovery is None:
        raise RecoveryError("not_recoverable", "This experiment was not submitted as recoverable")
    owner = recovery.owner
    own = (
        owner.pid == os.getpid()
        and process_identity_presence(owner.pid, owner.start_marker) is ProcessPresence.PRESENT
    )
    token_matches = (
        control_token is not None
        and bool(job.control_token)
        and secrets.compare_digest(control_token, job.control_token)
    )
    if not own and not token_matches:
        raise RecoveryError("resume_not_authorized", "The experiment control token is required")
    return job.control_token


async def resume_control_token(state: SessionState, job_id: str, control_token: str | None) -> str:
    """Authorize a private detached handoff from a fresh parent record."""
    job = await asyncio.to_thread(
        experiment_store.load_job, job_id, state.working_dir, own_is_alive=True
    )
    if job is None:
        raise JobNotFoundError(f"Experiment job {job_id} was not found")
    return authorize_resume(job, control_token)


def _current_owner() -> ProcessIdentity:
    return ProcessIdentity(os.getpid(), process_start_marker(os.getpid()))


def _required_recovery(job: ExperimentJob) -> JobRecovery:
    if job.recovery is None:
        raise RecoveryError("not_recoverable", "This experiment has no recovery contract")
    return job.recovery


def _required_journal(store: Store, root_request_id: str) -> RecoveryJournal:
    journal = load_journal(store, root_request_id)
    if journal is None:
        raise RecoveryError(
            "recovery_journal_missing", "Authoritative recovery journal is missing"
        )
    return journal


def _entry_for(journal: RecoveryJournal, job_id: str) -> JournalEntry:
    for entry in (journal.root, *journal.resumes.values()):
        if entry.job_id == job_id:
            return entry
    raise RecoveryError("recovery_record_invalid", "Job is not an admitted lineage attempt")


def _load_attempt(journal: RecoveryJournal, entry: JournalEntry, store: Store) -> ExperimentJob:
    if entry.job_id is None:
        raise RecoveryError("recovery_record_invalid", "No-op entry has no job record")
    job = experiment_store.load_job(entry.job_id, store.working_dir, own_is_alive=True)
    if job is None:
        # An unreadable record is evidence too. Never replace it with an older
        # candidate that would erase a launch or a completion.
        if os.path.lexists(store.job_record(entry.job_id)) or entry.phase != "prepared":
            raise RecoveryError("recovery_record_missing", "Recorded attempt cannot be loaded")
        job = candidate_job(entry, store)
    recovery = _required_recovery(job)
    if (
        job.request_id != entry.request_id
        or job.fingerprint != entry.fingerprint
        or recovery.attempt_index != entry.attempt_index
        or recovery.parent_job_id != entry.parent_job_id
        or recovery.root_job_id != journal.root_job_id
        or recovery.root_request_id != journal.root_request_id
    ):
        raise RecoveryError("recovery_record_invalid", "Attempt disagrees with its journal entry")
    return job


def persist_reconciled(job: ExperimentJob, store: Store) -> None:
    """Re-read an interrupted attempt under its lineage lock before saving it."""
    recovery = _required_recovery(job)
    with file_lock(store.recovery_lock(recovery.root_request_id)):
        journal = _required_journal(store, recovery.root_request_id)
        entry = _entry_for(journal, job.job_id)
        fresh = _load_attempt(journal, entry, store)
        if entry.phase == "prepared":
            candidate = candidate_job(entry, store)
            if _required_recovery(candidate).owner != _required_recovery(fresh).owner:
                # Adoption committed a new owner before updating the derived
                # record. A reader must not write its previous owner's view.
                return
        if fresh.restart_reconciled:
            experiment_store.save_job(fresh)


def _simulator_for(job: ExperimentJob, state: SessionState) -> type:
    candidates = [
        sim
        for sim in (*state.available_simulators.values(), *state.named_simulators.values())
        if sim.__name__ == job.simulator
    ]
    if not candidates:
        raise RecoveryError("recovery_simulator_unavailable", "Recorded simulator is unavailable")
    execution = _required_recovery(job).execution
    for simulator in candidates:
        current = executable_identity(simulator)
        if (
            current is not None
            and same_executable(execution.executable, current)
            and simulator_command(simulator, current) == execution.simulator_argv
        ):
            return simulator
    raise RecoveryError("recovery_execution_changed", "Recorded execution policy has changed")


def _verify_execution(job: ExperimentJob, store: Store, simulator: type) -> None:
    recovery = _required_recovery(job)
    execution = recovery.execution
    verify_execution_policy(execution, simulator)
    if job.output_folder is None:
        raise RecoveryError("recovery_record_invalid", "Lineage artifact directory is missing")
    root = store.lineage_run_dir(recovery.root_job_id, job.output_folder, simulator)
    verify_startup(execution.startup, root)
    for case in job.cases:
        if case.recovery is None:
            raise RecoveryError("recovery_record_invalid", "Case has no frozen input snapshot")
        if case.recovery.inputs.lineage_root != root:
            raise RecoveryError("recovery_record_invalid", "Case belongs to a different lineage")
        verify_case_inputs(
            case.recovery.inputs,
            native=case.native_statistics is not None,
            seeded=execution.simulator_seed is not None,
        )
        verify_seeded_driver(case, execution, root)
        if case.status == "produced":
            outputs = case.recovery.attempt.outputs
            if outputs is None:
                raise RecoveryError("recovery_artifact_missing", "Produced case has no digests")
            verify_produced_artifacts(outputs, root)


def _verify_absence(job: ExperimentJob, simulator: type) -> None:
    from ltspice_mcp.lib import wsl

    owner = _required_recovery(job).owner
    presence = process_identity_presence(owner.pid, owner.start_marker)
    if presence is ProcessPresence.UNKNOWN:
        raise RecoveryError("recovery_owner_unknown", "Prior owner could not be identified")
    # A live launcher can still spawn after an empty OS process scan, even if
    # its coordinator already timed out and retained the launch permit.
    if presence is ProcessPresence.PRESENT and any(
        case.failure_code == "kill_unconfirmed" for case in job.cases
    ):
        raise RecoveryError(
            "recovery_owner_active", "Prior owner may still launch or supervise a run"
        )
    names = simulator_executable_names(simulator)
    for case in job.cases:
        if case.recovery is None or case.recovery.attempt.launch is None:
            continue
        presence = simulator_presence(case.run_token, names)
        if presence is not ProcessPresence.ABSENT:
            raise RecoveryError(
                "recovery_process_active", "Prior simulator absence is unconfirmed"
            )
        if (
            "ltspice" in job.simulator.lower()
            and wsl.is_wsl()
            and wsl.windows_ltspice_presence(case.run_token) is not ProcessPresence.ABSENT
        ):
            raise RecoveryError(
                "recovery_process_active", "Windows simulator absence is unconfirmed"
            )


def _persist_discovery(job: ExperimentJob, store: Store, *, root: bool) -> None:
    experiment_store.save_job(job)
    if root:
        experiment_store.save_request_index(
            request_id=job.request_id,
            fingerprint=job.fingerprint,
            canonicalizer_version=job.canonicalizer_version,
            job_id=job.job_id,
            working_dir=store.working_dir,
        )
    # Resume request IDs belong to their lineage, not the global submission
    # namespace. They must not overwrite unrelated root request indexes.
    experiment_store.register_circuits(job, store.working_dir)


async def _replay_attempt(
    state: SessionState,
    store: Store,
    journal: RecoveryJournal,
    entry: JournalEntry,
    simulator: type,
) -> AdmissionResult:
    from ltspice_mcp.lib.experiment_runner import AdmissionResult

    job = await asyncio.to_thread(_load_attempt, journal, entry, store)
    if entry.phase == "launched":
        if job.restart_reconciled:
            await asyncio.to_thread(experiment_store.save_job, job)
        return AdmissionResult(job, replayed=True, start=False)
    # The prepared journal owns admission. A delayed derived record can still
    # name the previous owner after an adoption's journal write succeeds.
    candidate = await asyncio.to_thread(candidate_job, entry, store)
    recovery = _required_recovery(candidate)
    owner = recovery.owner
    presence = await asyncio.to_thread(process_identity_presence, owner.pid, owner.start_marker)
    if presence is not ProcessPresence.ABSENT:
        # Another admission may be between its durable barrier and scheduler.
        # Only a proven-dead owner licenses reconstructing and starting it.
        return AdmissionResult(job, replayed=True, start=False)
    if state.runners.get_experiment_runner_for(job) is not None:
        raise RecoveryError("recovery_owner_active", "A coordinator still owns this attempt")
    await asyncio.to_thread(_verify_absence, job, simulator)
    await asyncio.to_thread(_verify_execution, candidate, store, simulator)
    candidate.owner_pid = os.getpid()
    candidate.recovery = replace(recovery, owner=_current_owner())
    replacement = await asyncio.to_thread(prepared_entry, candidate)
    updated = (
        replace(journal, root=replacement)
        if entry.parent_job_id is None
        else replace(
            journal, resumes={**journal.resumes, path_digest(entry.request_id): replacement}
        )
    )
    await asyncio.to_thread(save_journal, store, updated)
    try:
        await asyncio.to_thread(
            _persist_discovery, candidate, store, root=entry.parent_job_id is None
        )
    except Exception as exc:
        raise _committed_failure(candidate, exc, replayed=True) from exc
    return AdmissionResult(candidate, replayed=True, start=True)


def _capture_execution(
    runner: ExperimentRunner, request: ExperimentRunRequest, job: ExperimentJob
) -> None:
    from ltspice_mcp.lib.experiment_runner import effective_run_timeout

    simulator = runner.simulator_class
    store = Store(request.state.working_dir)
    if request.simulator_seed is not None and any(case.native_statistics for case in job.cases):
        raise RecoveryError(
            "recovery_seed_unsupported", "Explicit seed cannot mix with native statistics"
        )
    if any(case.status != "queued" or case.recovery is None for case in job.cases):
        raise RecoveryError(
            "recovery_input_unavailable", "Every recoverable case needs validated frozen inputs"
        )
    prepare_native_cases(job, request.state.working_dir, simulator)
    if any(case.status != "queued" for case in job.cases):
        raise RecoveryError(
            "recovery_input_unavailable", "Native preparation failed before admission"
        )
    executable = request.simulator_executable
    if executable is None:
        raise RecoveryError("recovery_execution_unknown", "Simulator identity is incomplete")
    startup = prepare_startup(
        store, job.job_id, simulator, ini_source=request.state.config.ltspice_ini
    )
    timeout, source = effective_run_timeout(request)
    try:
        execution = ExecutionRecord(
            timeout,
            source or "unbounded",
            runner.case_capacity(request),
            request.job_deadline_s,
            request.kill_grace_s,
            simulator_command(simulator, executable),
            executable,
            getattr(simulator, "_compatibility_mode", None),
            sys.platform,
            startup,
            LAUNCH_POLICY if any(case.native_statistics for case in job.cases) else None,
            simulator_seed=request.simulator_seed,
        )
    except ValueError as exc:
        raise RecoveryError("recovery_execution_unknown", str(exc)) from exc
    job.recovery = JobRecovery(job.job_id, job.request_id, None, 0, _current_owner(), execution)
    sources = {source.circuit: source for source in job.sources}
    for case in job.cases:
        assert case.recovery is not None
        case.recovery = replace(
            case.recovery,
            inputs=capture_case_inputs(
                case,
                sources[case.circuit],
                lineage_root=case.recovery.inputs.lineage_root,
                seeded=execution.simulator_seed is not None,
            ),
        )
        prepare_seeded_driver(case, execution, store, job.job_id, simulator)
        prepare_case_startup(case, execution, store, job.job_id, simulator)
    _verify_execution(job, store, simulator)


async def admit_initial(
    runner: ExperimentRunner, request: ExperimentRunRequest
) -> AdmissionResult:
    """Commit the authoritative root before any derived records or launch."""
    from ltspice_mcp.lib.experiment_runner import (
        AdmissionResult,
        IdempotencyConflictError,
        request_gate,
        verify_replay,
    )

    store = Store(request.state.working_dir)
    async with (
        async_file_lock(store.recovery_lock(request.request_id)),
        request_gate(store.request_lock(request.request_id), request.request_id),
    ):
        journal = await asyncio.to_thread(load_journal, store, request.request_id)
        if journal is not None:
            if journal.root.fingerprint != request.fingerprint:
                raise IdempotencyConflictError("request_id already names a different payload")
            job = await asyncio.to_thread(_load_attempt, journal, journal.root, store)
            await asyncio.to_thread(
                verify_replay, job, request.request_id, request.simulator_executable
            )
            return await _replay_attempt(
                request.state, store, journal, journal.root, runner.simulator_class
            )
        lookup = await asyncio.to_thread(runner.read_request_index, request)
        if lookup.existing is not None or lookup.dangling:
            raise IdempotencyConflictError("Recoverable request has no authoritative journal")
        try:
            candidate = runner.materialize_job(request, await request.stage())
            await asyncio.to_thread(_capture_execution, runner, request, candidate)
            journal = await asyncio.to_thread(new_root_journal, candidate)
            # Reserved before the journal makes the id findable, so a replay of
            # it gets this job rather than a copy whose events are never set.
            request.state.job_registry.reserve(candidate)
            try:
                await asyncio.to_thread(save_journal, store, journal)
            except BaseException:
                request.state.job_registry.release(candidate)
                raise
        except Exception:
            await asyncio.to_thread(runner.discard_staged_run_dir, request, store.working_dir)
            raise
        try:
            await asyncio.to_thread(_persist_discovery, candidate, store, root=True)
        except Exception as exc:
            raise _committed_failure(candidate, exc) from exc
        return AdmissionResult(candidate, replayed=False, start=True)


async def mark_attempt_launched(job: ExperimentJob, state: SessionState) -> None:
    """Forbid prelaunch reconstruction before persisting any executable intent."""
    recovery = _required_recovery(job)
    store = Store(state.working_dir)
    async with async_file_lock(store.recovery_lock(recovery.root_request_id)):
        journal = await asyncio.to_thread(_required_journal, store, recovery.root_request_id)
        updated = mark_launched(journal, job.job_id)
        if updated != journal:
            await asyncio.to_thread(save_journal, store, updated)


def _eligible(case: ExperimentCase, *, retry_failed: bool, retry_cancelled: bool) -> bool:
    if case.recovery is None or case.status == "produced":
        return False
    if case.failure_code in {"server_restarted", "job_deadline"}:
        return True
    if case.status == "cancelled" or case.failure_code == "cancelled":
        return retry_cancelled
    return case.status == "failed" and retry_failed


def _make_child(
    parent: ExperimentJob,
    request_id: str,
    fingerprint: str,
    selected: set[str],
    store: Store,
    simulator: type,
) -> ExperimentJob:
    recovery = _required_recovery(parent)
    child = experiment_store.deserialize_job(
        experiment_store.serialize_job(parent), parent.store_path, liveness=OwnerLiveness.ALIVE
    )
    child.job_id = generate_id("exp")
    child.request_id, child.fingerprint = request_id, fingerprint
    child.control_token = secrets.token_urlsafe(32)
    child.store_path = store.job_record(child.job_id)
    child.owner_pid = os.getpid()
    child.recovery = replace(
        recovery,
        parent_job_id=parent.job_id,
        attempt_index=recovery.attempt_index + 1,
        owner=_current_owner(),
    )
    child.status, child.started_at, child.completed_at = "queued", now(), None
    child.error, child.restart_reconciled = None, False
    child.runs_done_event.clear()
    child.done_event.clear()
    child.observations, child.artifacts = [], []
    child.analysis = AnalysisStage(
        status="pending" if parent.analysis.request is not None else "not_requested",
        request=parent.analysis.request,
    )
    for case in child.cases:
        assert case.recovery is not None
        if case.case_id not in selected:
            if case.status == "produced":
                case.recovery = replace(
                    case.recovery, attempt=replace(case.recovery.attempt, reused=True)
                )
            continue
        case.run_token = f"{child.job_id}_case_{case.run_index}"
        case.recovery = replace(
            case.recovery,
            attempt=CaseAttempt(child.job_id, child.recovery.attempt_index, case.run_token),
        )
        case.status = "queued"
        case.raw_file = case.log_file = None
        case.error = case.failure_code = None
        case.failure_evidence = None
        case.submitted_at = case.completed_at = None
        case.simulator_version = None
        case.observations = []
        prepare_seeded_driver(case, recovery.execution, store, recovery.root_job_id, simulator)
        prepare_case_startup(case, recovery.execution, store, recovery.root_job_id, simulator)
        if case.native_statistics is not None:
            prepare_native_retry(
                case, store=store, root_job_id=recovery.root_job_id, simulator=simulator
            )
            source = next(source for source in child.sources if source.circuit == case.circuit)
            case.recovery = replace(
                case.recovery,
                inputs=capture_case_inputs(
                    case, source, lineage_root=case.recovery.inputs.lineage_root
                ),
            )
    child.failures = [
        failure_row(case)
        for case in child.cases
        if case.status in {"failed", "cancelled", "skipped"}
    ]
    child.completeness.recount(child.cases, execution_job_id=child.job_id)
    _verify_execution(child, store, simulator)
    return child


async def _start_attempt(
    state: SessionState,
    simulator: type,
    barrier: AdmissionResult,
    analysis_callback: AnalysisCallback | None,
) -> ExperimentReceipt:
    from ltspice_mcp.lib.experiment_runner import (
        ExperimentRunRequest,
        already_staged,
    )

    job = barrier.job
    execution = _required_recovery(job).execution
    folder = await asyncio.to_thread(Store(state.working_dir).runs_root, simulator)
    runner = state.runners.get_experiment_runner(
        asyncio.get_running_loop(), simulator, folder, max_parallel=state.config.max_parallel_sims
    )
    request = ExperimentRunRequest(
        state=state,
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        simulator=job.simulator,
        stage=already_staged,
        job_id=job.job_id,
        simulator_executable=execution.executable,
        recoverable=True,
        simulator_seed=execution.simulator_seed,
        max_parallel=execution.max_parallel,
        run_timeout_s=execution.run_timeout_s,
        job_deadline_s=execution.job_deadline_s,
        kill_grace_s=execution.kill_grace_s,
        analysis_request=job.analysis.request,
        analysis_callback=analysis_callback if job.analysis.request is not None else None,
    )
    ready = asyncio.get_running_loop().create_future()
    try:
        await runner.start_committed(request, barrier, ready)
    except Exception as exc:
        raise _committed_failure(job, exc, replayed=barrier.replayed) from exc
    return ready.result()


async def resume_experiment(
    state: SessionState,
    *,
    job_id: str,
    resume_request_id: str,
    control_token: str | None = None,
    case_ids: list[str] | None = None,
    retry_failed: bool = False,
    retry_cancelled: bool = False,
    analysis_callback: AnalysisCallback | None = None,
) -> ResumeReceipt:
    """Append one authorized attempt, retaining the original successful cases."""
    from ltspice_mcp.lib.experiment_runner import (
        AdmissionResult,
        IdempotencyConflictError,
        canonical_fingerprint,
    )

    store = Store(state.working_dir)
    if not state.config.persist_jobs or not state.job_registry.persist_enabled:
        raise RecoveryError(
            "recovery_persistence_required", "Resume requires durable job persistence"
        )
    parent = await asyncio.to_thread(
        experiment_store.load_job, job_id, state.working_dir, own_is_alive=True
    )
    if parent is None:
        raise JobNotFoundError(f"Experiment job {job_id} was not found")
    recovery = _required_recovery(parent)
    fingerprint = canonical_fingerprint(
        {
            "parent_job_id": job_id,
            "case_ids": sorted(set(case_ids)) if case_ids is not None else None,
            "retry_failed": retry_failed,
            "retry_cancelled": retry_cancelled,
        }
    )
    if not resume_request_id:
        raise RecoveryError("invalid_resume_request", "resume_request_id is required")
    async with async_file_lock(store.recovery_lock(recovery.root_request_id)):
        journal = await asyncio.to_thread(_required_journal, store, recovery.root_request_id)
        parent = await asyncio.to_thread(
            _load_attempt, journal, _entry_for(journal, job_id), store
        )
        authorize_resume(parent, control_token)
        entry = journal.resumes.get(path_digest(resume_request_id))
        if entry is not None:
            if entry.fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    "resume_request_id already names a different payload"
                )
            if entry.phase == "noop":
                return ResumeReceipt(parent, True, None, False)
            simulator = await asyncio.to_thread(_simulator_for, parent, state)
            barrier = await _replay_attempt(state, store, journal, entry, simulator)
        else:
            simulator = await asyncio.to_thread(_simulator_for, parent, state)
            if journal.head_job_id != parent.job_id:
                raise RecoveryError(
                    "recovery_stale_parent", "Resume must address the current lineage head"
                )
            if (
                not parent.done_event.is_set()
                or state.runners.get_experiment_runner_for(parent) is not None
            ):
                raise RecoveryError("recovery_owner_active", "The prior attempt is still active")
            await asyncio.to_thread(_verify_absence, parent, simulator)
            await asyncio.to_thread(_verify_execution, parent, store, simulator)
            requested = set(case_ids) if case_ids is not None else None
            known = {case.case_id for case in parent.cases}
            if requested is not None and (
                requested - known
                or any(
                    case.case_id in requested and case.status == "produced"
                    for case in parent.cases
                )
            ):
                raise RecoveryError(
                    "recovery_case_selection", "Selection contains unknown or produced cases"
                )
            selected = {
                case.case_id
                for case in parent.cases
                if (requested is None or case.case_id in requested)
                and _eligible(case, retry_failed=retry_failed, retry_cancelled=retry_cancelled)
            }
            if not selected:
                await asyncio.to_thread(
                    save_journal, store, add_noop(journal, resume_request_id, fingerprint, parent)
                )
                return ResumeReceipt(parent, False, None, False)
            child = await asyncio.to_thread(
                _make_child, parent, resume_request_id, fingerprint, selected, store, simulator
            )
            await asyncio.to_thread(
                save_journal, store, add_child(journal, resume_request_id, fingerprint, child)
            )
            try:
                await asyncio.to_thread(_persist_discovery, child, store, root=False)
            except Exception as exc:
                raise _committed_failure(child, exc) from exc
            barrier = AdmissionResult(child, replayed=False, start=True)
    receipt = await _start_attempt(state, simulator, barrier, analysis_callback)
    return ResumeReceipt(receipt.job, receipt.replayed, receipt.control_token, True)


async def lookup_root_recovery(
    state: SessionState,
    *,
    request_id: str,
    fingerprint: str,
    simulator: type,
    recoverable: bool,
    analysis_callback: AnalysisCallback | None = None,
) -> ExperimentReceipt | None:
    """Find a journal-only root before the tool inspects authoring paths."""
    from ltspice_mcp.lib.experiment_runner import (
        ExperimentRunRequest,
        IdempotencyConflictError,
        already_staged,
    )

    store = Store(state.working_dir)
    journal = await asyncio.to_thread(load_journal, store, request_id)
    if journal is None:
        return None
    if not recoverable or journal.root.fingerprint != fingerprint:
        raise IdempotencyConflictError(
            "request_id already belongs to a different recoverable request"
        )
    folder = await asyncio.to_thread(store.runs_root, simulator)
    runner = state.runners.get_experiment_runner(
        asyncio.get_running_loop(), simulator, folder, max_parallel=state.config.max_parallel_sims
    )
    request = ExperimentRunRequest(
        state=state,
        request_id=request_id,
        fingerprint=fingerprint,
        simulator=simulator.__name__,
        stage=already_staged,
        simulator_executable=await asyncio.to_thread(executable_identity, simulator),
        recoverable=True,
    )
    barrier = await admit_initial(runner, request)
    return await _start_attempt(state, simulator, barrier, analysis_callback)
