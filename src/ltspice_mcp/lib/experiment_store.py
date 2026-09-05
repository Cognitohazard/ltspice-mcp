"""Reading and writing the experiment job record.

The record's shape lives here; where it lives on disk is ``lib/store.py``, and
every path below comes from a :class:`~ltspice_mcp.lib.store.Store`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import now, parse_iso_datetime
from ltspice_mcp.lib.experiment_types import (
    TERMINAL_CASE_STATUSES,
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
    failure_row,
)
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.job_lifecycle import reconcile_experiment_restart, runs_terminal
from ltspice_mcp.lib.raw_parser import has_valid_raw_header
from ltspice_mcp.lib.store import (
    KIND_CANCELLATION,
    KIND_CIRCUIT_INDEX,
    KIND_EXPERIMENT,
    KIND_REQUEST_INDEX,
    OwnerLiveness,
    Store,
    accept,
    atomic_write_json,
    envelope,
    owner_liveness,
    owner_unknown_observation,
    pid_of,
    validate_job_id,
)

logger = logging.getLogger(__name__)

CANONICALIZER_VERSION = 3
# How a request's identity was computed — NOT a storage schema version. It says
# which fields the canonical fingerprint covers, so a reused ``request_id``
# whose record was hashed under an older definition raises the loud idempotency
# conflict instead of silently mis-comparing fingerprints.
#   2: execution.wait_s left the fingerprint (the dwell bounds only the
#      response, so a different dwell is the same experiment).
#   3: the measurements recipe gained histogram_bins, which participates
#      (asking for bins computes something new), so a request carrying that
#      recipe hashes different bytes than it did before the field existed.

_LIVE_STATUSES = frozenset({"queued", "running", "analyzing"})
_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
        "interrupted",
    }
)


def cancellation_requested(job_id: str, working_dir: Path) -> bool:
    """Whether an authorized durable cancellation marker exists."""
    return Store(working_dir).cancellation(job_id).is_file()


def cancel_authorized(job: ExperimentJob, control_token: str | None) -> bool:
    """Whether this process or the supplied control token may cancel ``job``."""
    return job.owner_pid == os.getpid() or (
        control_token is not None
        and bool(job.control_token)
        and secrets.compare_digest(control_token, job.control_token)
    )


def request_cancellation(
    job_id: str,
    working_dir: Path,
    control_token: str,
) -> ExperimentJob | None:
    """Authorize and durably request cancellation under the submission gate.

    The marker contains no authority token. The latest persisted coordinator
    record is re-read while the cross-process gate is held so a stale in-memory
    view cannot authorize cancellation.
    """
    store = Store(working_dir)
    with file_lock(store.cancellation_lock(job_id)):
        job = load_job(job_id, working_dir, own_is_alive=True)
        if job is None:
            return None
        if not cancel_authorized(job, control_token):
            raise PermissionError(f"Cancellation is not authorized for experiment job {job_id}")
        store.ensure_root()
        atomic_write_json(
            store.cancellation(job_id),
            envelope(
                KIND_CANCELLATION,
                job_id=job_id,
                requested_at=now().isoformat(),
            ),
        )
        return job


def _path_or_none(value: Any) -> Path | None:
    if value is None or str(value) in ("", "."):
        return None
    return Path(str(value))


def serialize_job(job: ExperimentJob) -> dict[str, Any]:
    """Return the durable JSON shape for an experiment coordinator."""
    cases: list[dict[str, Any]] = []
    for case in job.cases:
        cases.append(
            {
                "case_id": case.case_id,
                "run_index": case.run_index,
                "circuit": case.circuit,
                "circuit_path": str(case.circuit_path),
                "staged_deck": str(case.staged_deck),
                "deck_sha256": case.deck_sha256,
                "assignments": case.assignments,
                "status": case.status,
                "raw_file": str(case.raw_file) if case.raw_file else None,
                "log_file": str(case.log_file) if case.log_file else None,
                "error": case.error,
                "failure_code": case.failure_code,
                "failure_evidence": case.failure_evidence,
                "observations": case.observations,
                "submitted_at": case.submitted_at.isoformat() if case.submitted_at else None,
                "completed_at": case.completed_at.isoformat() if case.completed_at else None,
                "run_token": case.run_token,
                "step_index": case.step_index,
                "step_values": case.step_values,
            }
        )

    sources: list[dict[str, Any]] = []
    for source in job.sources:
        sources.append(
            {
                "circuit": source.circuit,
                "path": str(source.path),
                "sha256": source.sha256,
                "staged_deck": str(source.staged_deck),
                "manifest": [asdict(entry) for entry in source.manifest],
                "linter_version": source.linter_version,
                "simulator": source.simulator,
                "dialect": source.dialect,
                "lint_findings": source.lint_findings,
            }
        )

    analysis = {
        "status": job.analysis.status,
        "request": job.analysis.request,
        "result": job.analysis.result,
        "error": job.analysis.error,
        "started_at": job.analysis.started_at.isoformat() if job.analysis.started_at else None,
        "completed_at": (
            job.analysis.completed_at.isoformat() if job.analysis.completed_at else None
        ),
        "observations": job.analysis.observations,
    }
    return envelope(
        KIND_EXPERIMENT,
        job_id=job.job_id,
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        canonicalizer_version=job.canonicalizer_version,
        control_token=job.control_token,
        pid=job.owner_pid,
        simulator=job.simulator,
        status=job.status,
        started_at=job.started_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        output_folder=str(job.output_folder) if job.output_folder else None,
        completeness=asdict(job.completeness),
        cases=cases,
        sources=sources,
        failures=job.failures,
        observations=job.observations,
        artifacts=job.artifacts,
        analysis=analysis,
    )


def save_job(job: ExperimentJob) -> Path:
    """Persist an experiment at its validated working-store path."""
    validate_job_id(job.job_id)
    root = job.store_path.parent.resolve()
    expected = (root / f"{job.job_id}.json").resolve()
    if job.store_path.resolve() != expected or expected.parent != root:
        raise ValueError(f"Experiment store path is not contained: {job.store_path}")
    job.store_path = expected
    atomic_write_json(expected, serialize_job(job))
    logger.debug("Persisted experiment job %s to %s", job.job_id, expected)
    return expected


def save_request_index(
    *,
    request_id: str,
    fingerprint: str,
    canonicalizer_version: int,
    job_id: str,
    working_dir: Path,
) -> Path:
    """Write the durable request-id mapping."""
    validate_job_id(job_id)
    store = Store(working_dir)
    store.ensure_root()
    path = store.request_index(request_id)
    atomic_write_json(
        path,
        envelope(
            KIND_REQUEST_INDEX,
            request_id=request_id,
            fingerprint=fingerprint,
            canonicalizer_version=canonicalizer_version,
            job_id=job_id,
            created_at=now().isoformat(),
        ),
    )
    return path


def load_request_index(request_id: str, working_dir: Path) -> dict[str, Any] | None:
    """Load the exact request-id index entry, rejecting hash collisions."""
    path = Store(working_dir).request_index(request_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable experiment request index %s: %s", path, exc)
        return None
    if not accept(data, path, kind=KIND_REQUEST_INDEX, log=logger):
        return None
    if data.get("request_id") != request_id:
        logger.warning("Ignoring mismatched experiment request index %s", path)
        return None
    return data


def register_circuits(job: ExperimentJob, working_dir: Path) -> list[Path]:
    """Index this job under every distinct circuit it ran, in the working store.

    This replaces the pointer file that used to be written next to each source
    circuit and named an absolute path back into the store. That path had to be
    re-validated on every read (a tampered one could name a record outside the
    store), and it could only ever resolve for the session whose working
    directory it happened to name — so a pointer beside a circuit was already
    unusable from anywhere else. An index inside the store carries a job id and
    nothing else: there is no path to validate, and discovery is scoped to the
    store that owns the records, which is where it always actually was.
    """
    store = Store(working_dir)
    store.ensure_root()
    saved: list[Path] = []
    seen: set[Path] = set()
    for source in job.sources:
        try:
            circuit = source.path.resolve()
        except OSError:
            circuit = source.path
        if circuit in seen:
            continue
        seen.add(circuit)
        path = store.circuit_index(circuit, job.job_id)
        atomic_write_json(
            path,
            envelope(
                KIND_CIRCUIT_INDEX,
                job_id=job.job_id,
                circuit=str(circuit),
                created_at=now().isoformat(),
            ),
        )
        saved.append(path)
    return saved


def _manifest_entry(data: dict[str, Any]) -> ManifestEntry:
    return ManifestEntry(
        path=Path(str(data.get("path", ""))),
        sha256=str(data.get("sha256", "")),
        staged=bool(data.get("staged", False)),
        live=bool(data.get("live", False)),
        staged_path=_path_or_none(data.get("staged_path")),
        reason=data.get("reason"),
        section=data.get("section"),
    )


def _source_record(data: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        circuit=str(data.get("circuit", "")),
        path=Path(str(data.get("path", ""))),
        sha256=str(data.get("sha256", "")),
        staged_deck=Path(str(data.get("staged_deck", ""))),
        manifest=[
            _manifest_entry(item) for item in data.get("manifest", []) if isinstance(item, dict)
        ],
        linter_version=str(data.get("linter_version", "")),
        simulator=str(data.get("simulator", "")),
        dialect=data.get("dialect"),
        lint_findings=list(data.get("lint_findings") or []),
    )


def _case_record(data: dict[str, Any]) -> ExperimentCase:
    return ExperimentCase(
        case_id=str(data.get("case_id", "")),
        run_index=int(data.get("run_index", 0)),
        circuit=str(data.get("circuit", "")),
        circuit_path=Path(str(data.get("circuit_path", ""))),
        staged_deck=Path(str(data.get("staged_deck", ""))),
        deck_sha256=str(data.get("deck_sha256", "")),
        assignments=dict(data.get("assignments") or {}),
        status=str(data.get("status", "queued")),  # type: ignore[arg-type]
        raw_file=_path_or_none(data.get("raw_file")),
        log_file=_path_or_none(data.get("log_file")),
        error=data.get("error"),
        failure_code=data.get("failure_code"),
        failure_evidence=data.get("failure_evidence"),
        observations=list(data.get("observations") or []),
        submitted_at=parse_iso_datetime(data.get("submitted_at")),
        completed_at=parse_iso_datetime(data.get("completed_at")),
        run_token=str(data.get("run_token", "")),
        step_index=data.get("step_index"),
        step_values=dict(data.get("step_values") or {}),
    )


def _analysis_stage(data: dict[str, Any] | None) -> AnalysisStage:
    raw = data or {}
    return AnalysisStage(
        status=str(raw.get("status", "not_requested")),  # type: ignore[arg-type]
        request=raw.get("request"),
        result=raw.get("result"),
        error=raw.get("error"),
        started_at=parse_iso_datetime(raw.get("started_at")),
        completed_at=parse_iso_datetime(raw.get("completed_at")),
        observations=list(raw.get("observations") or []),
    )


def _produced_artifacts(job: ExperimentJob, case: ExperimentCase) -> tuple[Path, Path] | None:
    """A non-terminal case's raw/log pair when both verify on disk, else None.

    Case progress is checkpointed sparsely (every ``total // 20``-th event), so
    a crash can lose the terminal mark of a case that already wrote its results.
    Counting that as a shortfall is data loss dressed as accounting: the run
    happened and its artifacts are still there. The path is deterministic —
    the runner names every artifact ``{run_token}.{ext}`` inside the job's
    output folder, the same reconstruction ``_remove_case_artifacts`` uses to
    DELETE them — and the raw's header magic is what keeps a truncated or
    unrelated file from being promoted. Mirrors the legacy registry's
    ``has_valid_raw_header`` promotion for single-run jobs.

    Returns None (and the case stays a failure) for a record written before the
    output folder was persisted: the honest direction when the artifacts cannot
    be located at all.
    """
    if job.output_folder is None or not case.run_token:
        return None
    extension = ".qraw" if "qspice" in job.simulator.lower() else ".raw"
    raw = case.raw_file or job.output_folder / f"{case.run_token}{extension}"
    log = case.log_file or job.output_folder / f"{case.run_token}.log"
    if not has_valid_raw_header(raw):
        return None
    try:
        if not log.is_file():
            return None
    except OSError:
        return None
    return raw, log


def _reconcile_restart(job: ExperimentJob, *, liveness: OwnerLiveness) -> None:
    if job.status not in _LIVE_STATUSES:
        return
    if liveness is OwnerLiveness.UNKNOWN:
        # Not an answer, so not grounds to take a peer's live experiment away
        # from it. Keep the recorded status and say why it was not checked.
        job.observations.append(owner_unknown_observation(job.owner_pid))
        return
    if not liveness.is_dead:
        return
    observation = {
        "code": "server_restarted",
        "kind": "lifecycle",
        "detail": "The owning server stopped before the experiment reached terminality.",
    }
    job.observations.append(observation)
    runs_were_terminal = all(case.status in TERMINAL_CASE_STATUSES for case in job.cases)
    analysis_interrupted = job.analysis.status in {"pending", "running"}
    if analysis_interrupted:
        job.analysis = replace(
            job.analysis,
            status="failed",
            error="Server restarted before attached analysis completed",
            completed_at=now(),
            observations=[*job.analysis.observations, observation],
        )
    reconciled: list[ExperimentCase] = []
    abandoned: list[ExperimentCase] = []
    recovered: list[ExperimentCase] = []
    for case in job.cases:
        if case.status in TERMINAL_CASE_STATUSES:
            reconciled.append(case)
            continue
        artifacts = _produced_artifacts(job, case)
        if artifacts is not None:
            raw, log = artifacts
            promoted = replace(
                case,
                status="produced",
                raw_file=raw,
                log_file=log,
                completed_at=case.completed_at or now(),
            )
            recovered.append(promoted)
            reconciled.append(promoted)
            continue
        failed = replace(
            case,
            status="failed",
            failure_code="server_restarted",
            # Name the mechanism the caller can act on — the owning process
            # exited — never "the server": the store cannot see which interface
            # owned the job, and for the in-process API the owner is the
            # caller's own script (a wait=False submission from a process
            # that exits leaves exactly this shape).
            error=(
                "The owning process exited before this case reached "
                "terminality, leaving the run unsupervised; keep the "
                "submitting process (or a long-lived server) alive until "
                "the job finishes"
            ),
            completed_at=now(),
        )
        abandoned.append(failed)
        reconciled.append(failed)
    job.cases = reconciled
    job.failures.extend(failure_row(case) for case in abandoned)
    if recovered:
        job.observations.append(
            {
                "code": "unpersisted_runs_recovered",
                "kind": "reconciliation",
                "detail": (
                    f"{len(recovered)} case(s) had written their results before the "
                    "server stopped but never recorded them; the artifacts were "
                    "verified on disk and the cases count as produced."
                ),
                "evidence": {"case_ids": [case.case_id for case in recovered]},
            }
        )
    job.completeness.recount(job.cases)
    if runs_terminal(job.status) or runs_were_terminal:
        has_failure = job.completeness.fell_short or job.analysis.status in {
            "failed",
            "cancelled",
        }
        reconcile_experiment_restart(
            job,
            "completed_with_failures" if has_failure else "completed",
        )
        return
    reconcile_experiment_restart(job, "interrupted")


def _deserialize_job(
    data: dict[str, Any],
    store_path: Path,
    *,
    own_is_alive: bool,
) -> ExperimentJob:
    started_at = parse_iso_datetime(data.get("started_at")) or now()
    completeness_data = data.get("completeness") or {}
    completeness = Completeness(
        declared=int(completeness_data.get("declared", 0)),
        expanded=int(completeness_data.get("expanded", 0)),
        submitted=int(completeness_data.get("submitted", 0)),
        produced=int(completeness_data.get("produced", 0)),
        failed=int(completeness_data.get("failed", 0)),
        cancelled=int(completeness_data.get("cancelled", 0)),
        skipped=int(completeness_data.get("skipped", 0)),
    )
    job = ExperimentJob(
        job_id=str(data["job_id"]),
        request_id=str(data.get("request_id", "")),
        fingerprint=str(data.get("fingerprint", "")),
        canonicalizer_version=int(data.get("canonicalizer_version", 0)),
        control_token=str(data.get("control_token", "")),
        store_path=store_path,
        cases=[_case_record(item) for item in data.get("cases", []) if isinstance(item, dict)],
        sources=[
            _source_record(item) for item in data.get("sources", []) if isinstance(item, dict)
        ],
        simulator=str(data.get("simulator", "")),
        completeness=completeness,
        status=str(data.get("status", "interrupted")),  # type: ignore[arg-type]
        started_at=started_at,
        completed_at=parse_iso_datetime(data.get("completed_at")),
        error=data.get("error"),
        output_folder=_path_or_none(data.get("output_folder")),
        failures=list(data.get("failures") or []),
        observations=list(data.get("observations") or []),
        artifacts=list(data.get("artifacts") or []),
        analysis=_analysis_stage(data.get("analysis")),
        owner_pid=pid_of(data) or 0,
    )
    _reconcile_restart(
        job,
        liveness=owner_liveness(pid_of(data), own_is_alive=own_is_alive),
    )
    if all(case.status in TERMINAL_CASE_STATUSES for case in job.cases):
        job.runs_done_event.set()
    if job.status in _TERMINAL_STATUSES:
        job.done_event.set()
    return job


def load_job_from_path(
    path: Path,
    working_dir: Path,
    *,
    own_is_alive: bool = False,
) -> ExperimentJob | None:
    """Load one coordinator record after validating store-root containment."""
    store = Store(working_dir)
    resolved = path.resolve()
    if resolved.parent != store.experiments_dir:
        raise ValueError(f"Experiment record is outside the working store: {path}")
    validate_job_id(resolved.stem)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable experiment job %s: %s", resolved, exc)
        return None
    if not accept(data, resolved, kind=KIND_EXPERIMENT, log=logger):
        return None
    if data.get("job_id") != resolved.stem:
        logger.warning("Skipping experiment record %s: job id does not match filename", resolved)
        return None
    try:
        return _deserialize_job(data, resolved, own_is_alive=own_is_alive)
    except Exception as exc:
        logger.warning("Skipping malformed experiment job %s: %s", resolved, exc)
        return None


def load_job(
    job_id: str,
    working_dir: Path,
    *,
    own_is_alive: bool = False,
) -> ExperimentJob | None:
    """Load a coordinator directly by validated job id."""
    return load_job_from_path(
        Store(working_dir).job_record(job_id),
        working_dir,
        own_is_alive=own_is_alive,
    )


def load_jobs_for_circuit(
    circuit_path: Path,
    working_dir: Path,
    prefer: Mapping[str, ExperimentJob] | None = None,
) -> tuple[list[ExperimentJob], list[dict[str, Any]]]:
    """Every experiment in this store that ran ``circuit_path``.

    ``prefer`` maps job ids to live registry instances (snapshotted on the
    event loop by the caller): a job this process owns is returned from
    there instead of disk, because its most recent transitions may still be
    in a pending fire-and-forget persist.

    An index entry whose record has gone (evicted, or deleted by hand) is not
    an anomaly and is skipped in silence; only an entry this build cannot read
    at all becomes an observation.
    """
    store = Store(working_dir)
    jobs: list[ExperimentJob] = []
    observations: list[dict[str, Any]] = []
    index_dir = store.circuit_index_dir(circuit_path)
    if not index_dir.is_dir():
        return jobs, observations
    for path in sorted(index_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not accept(data, path, kind=KIND_CIRCUIT_INDEX, log=logger):
                raise ValueError("unsupported experiment index record")
            job_id = validate_job_id(str(data.get("job_id", "")))
            live = prefer.get(job_id) if prefer else None
            if live is not None:
                jobs.append(live)
                continue
            job = load_job_from_path(
                store.job_record(job_id),
                working_dir,
                own_is_alive=True,
            )
            if job is not None:
                jobs.append(job)
        except Exception as exc:
            observation = {
                "code": "experiment_index_invalid",
                "kind": "discovery",
                "detail": f"Skipped experiment index entry {path}: {exc}",
                "evidence": {"entry": str(path)},
            }
            observations.append(observation)
            # Debug, not warning: the fact still reaches whoever asked — the
            # registry accumulates these and the jobs listing returns them —
            # so nothing is lost by keeping it out of a library's boot output.
            logger.debug(observation["detail"])
    return jobs, observations


def delete_job(job: ExperimentJob, working_dir: Path) -> None:
    """Delete a coordinator, its circuit index entries, and its request index."""
    store = Store(working_dir)
    for source in job.sources:
        try:
            store.circuit_index(source.path, job.job_id).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Could not remove experiment index entry for %s: %s", job.job_id, exc)

    with file_lock(store.request_lock(job.request_id)):
        index_path = store.request_index(job.request_id)
        index = load_request_index(job.request_id, working_dir)
        if index is not None and index.get("job_id") == job.job_id:
            with contextlib.suppress(FileNotFoundError):
                index_path.unlink()
        with file_lock(store.cancellation_lock(job.job_id)):
            with contextlib.suppress(FileNotFoundError):
                job.store_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                store.cancellation(job.job_id).unlink()
