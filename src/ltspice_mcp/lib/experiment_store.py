"""Working-directory persistence for multi-circuit experiment jobs."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
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
from ltspice_mcp.lib.store_common import (
    EXPERIMENT_JOB_SCHEMA,
    OwnerLiveness,
    accept_schema,
    atomic_write_json,
    owner_liveness,
    owner_unknown_observation,
    pid_of,
    schema_envelope,
)

logger = logging.getLogger(__name__)

SCHEMA = EXPERIMENT_JOB_SCHEMA
SCHEMA_VERSION = 2
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2})
# The legacy job store writes into this same directory; its records are the
# expected other half of the layout, not a corrupted file.


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Admit v1 records; their untagged analysis result remains legacy public data."""
    return data


_MIGRATIONS: dict[int, Any] = {1: _migrate_v1_to_v2}

POINTER_SCHEMA = "ltspice-mcp/experiment-pointer"
POINTER_SCHEMA_VERSION = 1
# Version 2: execution.wait_s left the canonical fingerprint (the dwell bounds
# only the response, so a different dwell is the same experiment). A reused
# request_id whose record was hashed under an older version raises the loud
# idempotency conflict instead of silently mis-comparing fingerprints.
# Version 3: the measurements recipe gained histogram_bins. It participates
# (asking for bins computes something new, like include.outliers does), so a
# request carrying that recipe now hashes different bytes than it did before
# the field existed — which is exactly the condition a version bump exists to
# report accurately instead of as "your arguments differ".
CANONICALIZER_VERSION = 3

SIDECAR_DIRNAME = ".ltspice-mcp"
JOBS_SUBDIR = "jobs"
EXPERIMENT_POINTERS_SUBDIR = "experiments"
REQUESTS_SUBDIR = "requests"
LOCKS_SUBDIR = "locks"
CANCELLATIONS_SUBDIR = "cancellations"

CANCELLATION_SCHEMA = "ltspice-mcp/experiment-cancellation"
CANCELLATION_SCHEMA_VERSION = 1

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
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


def validate_job_id(job_id: str) -> str:
    """Validate a server-generated job id before it reaches path construction."""
    if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(
            "Invalid job id: expected 1-64 letters, digits, underscores, or hyphens, "
            "starting with a letter or digit"
        )
    return job_id


def working_store_root(working_dir: Path) -> Path:
    """Resolved working-directory home for coordinator records."""
    return (working_dir / SIDECAR_DIRNAME / JOBS_SUBDIR).resolve()


def record_path(job_id: str, working_dir: Path) -> Path:
    """Contained coordinator-record path for ``job_id``."""
    validate_job_id(job_id)
    root = working_store_root(working_dir)
    candidate = (root / f"{job_id}.json").resolve()
    if candidate.parent != root:
        raise ValueError(f"Experiment job path escapes the working store: {job_id!r}")
    return candidate


def request_digest(request_id: str) -> str:
    """Filesystem-safe digest used for request indexes and their locks."""
    import hashlib

    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def request_index_path(request_id: str, working_dir: Path) -> Path:
    """Request-id index path; the raw request id never becomes a path segment."""
    return working_store_root(working_dir) / REQUESTS_SUBDIR / f"{request_digest(request_id)}.json"


def request_lock_target(request_id: str, working_dir: Path) -> Path:
    """Target whose ``file_lock`` sidecar is ``request-{digest}.lock``."""
    return working_store_root(working_dir) / LOCKS_SUBDIR / f"request-{request_digest(request_id)}"


def cancellation_path(job_id: str, working_dir: Path) -> Path:
    """Contained durable cancellation marker for one experiment."""
    validate_job_id(job_id)
    return working_store_root(working_dir) / CANCELLATIONS_SUBDIR / f"{job_id}.json"


def cancellation_lock_target(job_id: str, working_dir: Path) -> Path:
    """Cross-process gate shared by cancellation and case submission."""
    validate_job_id(job_id)
    return working_store_root(working_dir) / LOCKS_SUBDIR / f"cancel-{job_id}"


def cancellation_requested(job_id: str, working_dir: Path) -> bool:
    """Whether an authorized durable cancellation marker exists."""
    return cancellation_path(job_id, working_dir).is_file()


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
    with file_lock(cancellation_lock_target(job_id, working_dir)):
        job = load_job(job_id, working_dir, own_is_alive=True)
        if job is None:
            return None
        if not cancel_authorized(job, control_token):
            raise PermissionError(f"Cancellation is not authorized for experiment job {job_id}")
        atomic_write_json(
            cancellation_path(job_id, working_dir),
            schema_envelope(
                CANCELLATION_SCHEMA,
                CANCELLATION_SCHEMA_VERSION,
                kind="experiment_cancellation",
                job_id=job_id,
                requested_at=now().isoformat(),
            ),
        )
        return job


def pointer_dir(circuit_path: Path) -> Path:
    """Per-circuit directory holding experiment pointer records."""
    return circuit_path.parent / SIDECAR_DIRNAME / JOBS_SUBDIR / EXPERIMENT_POINTERS_SUBDIR


def pointer_path(circuit_path: Path, job_id: str) -> Path:
    validate_job_id(job_id)
    return pointer_dir(circuit_path) / f"{job_id}.json"


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
    return schema_envelope(
        SCHEMA,
        SCHEMA_VERSION,
        kind="experiment",
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
    path = request_index_path(request_id, working_dir)
    atomic_write_json(
        path,
        {
            "request_id": request_id,
            "fingerprint": fingerprint,
            "canonicalizer_version": canonicalizer_version,
            "job_id": job_id,
            "created_at": now().isoformat(),
        },
    )
    return path


def load_request_index(request_id: str, working_dir: Path) -> dict[str, Any] | None:
    """Load the exact request-id index entry, rejecting hash collisions."""
    path = request_index_path(request_id, working_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable experiment request index %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("request_id") != request_id:
        logger.warning("Ignoring mismatched experiment request index %s", path)
        return None
    return data


def save_pointers(job: ExperimentJob) -> list[Path]:
    """Write one lightweight pointer beside every distinct source circuit."""
    saved: list[Path] = []
    seen: set[Path] = set()
    target = job.store_path.resolve()
    for source in job.sources:
        try:
            circuit = source.path.resolve()
        except OSError:
            circuit = source.path
        if circuit in seen:
            continue
        seen.add(circuit)
        path = pointer_path(circuit, job.job_id)
        atomic_write_json(
            path,
            schema_envelope(
                POINTER_SCHEMA,
                POINTER_SCHEMA_VERSION,
                kind="experiment_pointer",
                job_id=job.job_id,
                target=str(target),
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
            # exited — never "the server": the store cannot see which door
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
    root = working_store_root(working_dir)
    resolved = path.resolve()
    if resolved.parent != root:
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
    if not isinstance(data, dict) or not accept_schema(
        data,
        resolved,
        schema=SCHEMA,
        current_version=SCHEMA_VERSION,
        supported_versions=SUPPORTED_VERSIONS,
        migrations=_MIGRATIONS,
        logger=logger,
    ):
        return None
    if data.get("kind") != "experiment":
        logger.warning("Skipping experiment record %s: wrong kind %r", resolved, data.get("kind"))
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
        record_path(job_id, working_dir),
        working_dir,
        own_is_alive=own_is_alive,
    )


def load_pointer_jobs(
    circuit_path: Path,
    working_dir: Path,
    prefer: Mapping[str, ExperimentJob] | None = None,
) -> tuple[list[ExperimentJob], list[dict[str, Any]]]:
    """Resolve a circuit's pointers to validated working-store records.

    ``prefer`` maps job ids to live registry instances (snapshotted on the
    event loop by the caller): a job this process owns is returned from
    there instead of disk, because its most recent transitions may still be
    in a pending fire-and-forget persist.
    """
    jobs: list[ExperimentJob] = []
    observations: list[dict[str, Any]] = []
    target_dir = pointer_dir(circuit_path)
    if not target_dir.is_dir():
        return jobs, observations
    root = working_store_root(working_dir)
    for path in sorted(target_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if (
                data.get("schema") != POINTER_SCHEMA
                or data.get("schema_version") != POINTER_SCHEMA_VERSION
                or data.get("kind") != "experiment_pointer"
            ):
                raise ValueError("unsupported experiment pointer schema")
            job_id = validate_job_id(str(data.get("job_id", "")))
            live = prefer.get(job_id) if prefer else None
            if live is not None:
                jobs.append(live)
                continue
            target_raw = data.get("target")
            if not isinstance(target_raw, str):
                raise TypeError("pointer target is missing")
            target = Path(target_raw).resolve()
            if target.parent != root or target.name != f"{job_id}.json":
                raise ValueError("pointer target is outside the configured working store")
            job = load_job_from_path(target, working_dir, own_is_alive=True)
            if job is not None:
                jobs.append(job)
        except Exception as exc:
            observation = {
                "code": "experiment_pointer_invalid",
                "kind": "discovery",
                "detail": f"Skipped experiment pointer {path}: {exc}",
                "evidence": {"pointer": str(path)},
            }
            observations.append(observation)
            # Debug, not warning: the pointer index is global, so a fresh
            # working directory reaches other projects' stale records and
            # narrates a dozen of them before the caller has done anything.
            # The fact still reaches whoever asked — the registry accumulates
            # these and the jobs listing returns them — so nothing is lost by
            # keeping them out of a library's boot output.
            logger.debug(observation["detail"])
    return jobs, observations


def delete_job(job: ExperimentJob, working_dir: Path) -> None:
    """Delete a coordinator, its pointers, and its still-current request index."""
    for source in job.sources:
        try:
            pointer_path(source.path, job.job_id).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Could not remove experiment pointer for %s: %s", job.job_id, exc)

    lock_target = request_lock_target(job.request_id, working_dir)
    with file_lock(lock_target):
        index_path = request_index_path(job.request_id, working_dir)
        index = load_request_index(job.request_id, working_dir)
        if index is not None and index.get("job_id") == job.job_id:
            with contextlib.suppress(FileNotFoundError):
                index_path.unlink()
        with file_lock(cancellation_lock_target(job.job_id, working_dir)):
            with contextlib.suppress(FileNotFoundError):
                job.store_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                cancellation_path(job.job_id, working_dir).unlink()
