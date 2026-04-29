"""Per-circuit JSON persistence for simulation and batch jobs.

Jobs are stored in ``{circuit_parent}/.ltspice-mcp/jobs/{job_id}.json`` so they
travel with the circuit they belong to. Writes are atomic (tempfile + rename).
Loads are lazy — the server only reads a circuit's sidecar directory the first
time a tool touches that circuit in a session.

Jobs whose server process died while they were running come back as
``interrupted`` (see ``_finalize_loaded_status``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import atomic_write_json, parse_iso_datetime
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    MonteCarloConfig,
    SimulationJob,
    SweepConfig,
    SweepDimension,
)

logger = logging.getLogger(__name__)

SIDECAR_DIRNAME = ".ltspice-mcp"
JOBS_SUBDIR = "jobs"
SCHEMA = "ltspice-mcp/job"
SCHEMA_VERSION = 1
# Versions this build can READ after applying ``_MIGRATIONS``. Always
# includes the current version; older versions are added once their
# migration function lands in ``_MIGRATIONS``.
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})
INTERRUPTED_STATUS = "interrupted"


def _migrate(data: dict, from_version: int) -> dict:
    """Upgrade a loaded record from ``from_version`` to ``SCHEMA_VERSION``.

    Applies each step in the chain ``_MIGRATIONS[v](data)``. When adding a
    new schema version, bump ``SCHEMA_VERSION``, add the current version to
    ``SUPPORTED_VERSIONS``, and register a migration function here.
    Migrations MUST be idempotent-safe: if called twice on the same dict
    they should not corrupt it.
    """
    current = from_version
    while current < SCHEMA_VERSION:
        migrate_fn = _MIGRATIONS.get(current)
        if migrate_fn is None:
            raise ValueError(
                f"No migration path from schema_version {current} to {SCHEMA_VERSION}"
            )
        data = migrate_fn(data)
        current += 1
    data["schema_version"] = SCHEMA_VERSION
    return data


# Registered migration functions. Key N transforms v(N) into v(N+1).
# Keep each function focused and reversible where possible.
_MIGRATIONS: dict[int, Any] = {}


def sidecar_dir(circuit_path: Path) -> Path:
    """Return the ``.ltspice-mcp/jobs`` directory next to a circuit file."""
    return circuit_path.parent / SIDECAR_DIRNAME / JOBS_SUBDIR


def _job_file(job_id: str, dir_: Path) -> Path:
    return dir_ / f"{job_id}.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def _serialize_sim_job(job: SimulationJob) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "kind": "simulation",
        "netlist": str(job.netlist),
        "simulator": job.simulator,
        "status": job.status,
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "raw_file": str(job.raw_file) if job.raw_file else None,
        "log_file": str(job.log_file) if job.log_file else None,
        "error": job.error,
    }


def _serialize_batch_job(job: BatchJob) -> dict:
    # dataclasses.asdict handles nested SweepDimension / MonteCarloConfig cleanly.
    sweep_cfg = asdict(job.sweep_config) if job.sweep_config else None
    mc_cfg = asdict(job.mc_config) if job.mc_config else None
    # run_results may contain Path objects inside values — coerce to str.
    run_results_clean: dict[str, dict[str, Any]] = {}
    for idx, res in job.run_results.items():
        run_results_clean[str(idx)] = {
            "raw_file": str(res["raw_file"]) if res.get("raw_file") else None,
            "log_file": str(res["log_file"]) if res.get("log_file") else None,
            "params": dict(res.get("params") or {}),
        }

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "kind": "batch",
        "job_type": job.job_type,
        "netlist": str(job.netlist),
        "total_runs": job.total_runs,
        "completed_runs": job.completed_runs,
        "failed_runs": job.failed_runs,
        "status": job.status,
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "run_results": run_results_clean,
        "sweep_config": sweep_cfg,
        "mc_config": mc_cfg,
    }


def serialize_job(job: SimulationJob | BatchJob) -> dict:
    """Return a JSON-ready dict for either job flavour."""
    if isinstance(job, SimulationJob):
        return _serialize_sim_job(job)
    return _serialize_batch_job(job)


def save_job(job: SimulationJob | BatchJob) -> Path:
    """Persist a job to its circuit's sidecar directory. Returns the file path."""
    target_dir = sidecar_dir(job.netlist)
    path = _job_file(job.job_id, target_dir)
    atomic_write_json(path, serialize_job(job), default=_json_default)
    logger.debug("Persisted job %s to %s", job.job_id, path)
    return path


def delete_job(job: SimulationJob | BatchJob) -> None:
    """Delete a job's persisted JSON file, if present."""
    path = _job_file(job.job_id, sidecar_dir(job.netlist))
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _finalize_loaded_status(raw_status: str) -> tuple[str, bool]:
    """Translate a loaded status.

    Returns (effective_status, was_interrupted). Running/queued jobs whose
    owning process is gone come back as ``interrupted``.
    """
    if raw_status in NON_TERMINAL_LIVE_STATUSES:
        return INTERRUPTED_STATUS, True
    return raw_status, False


def _accept_schema(data: dict, source: Path) -> bool:
    """Verify a loaded record's schema is one we understand, migrating if needed.

    Modifies ``data`` in place when applying a migration so callers get the
    current-schema shape without special-casing versions. Returns False for
    unsupported versions or schemas (caller should skip that record).
    """
    schema = data.get("schema")
    if schema != SCHEMA:
        logger.warning(
            "Skipping job file %s: unexpected schema %r (expected %s)",
            source,
            schema,
            SCHEMA,
        )
        return False

    raw_version = data.get("schema_version")
    if raw_version is None:
        logger.warning("Skipping job file %s: missing schema_version", source)
        return False
    if not isinstance(raw_version, int):
        logger.warning(
            "Skipping job file %s: schema_version must be an integer, got %r",
            source,
            raw_version,
        )
        return False

    if raw_version == SCHEMA_VERSION:
        return True
    if raw_version in SUPPORTED_VERSIONS and raw_version < SCHEMA_VERSION:
        try:
            _migrate(data, raw_version)
        except ValueError as e:
            logger.warning("Skipping job file %s: %s", source, e)
            return False
        return True

    logger.warning(
        "Skipping job file %s: unsupported schema_version %d (this build reads %s)",
        source,
        raw_version,
        sorted(SUPPORTED_VERSIONS),
    )
    return False


def _deserialize_sim_job(data: dict) -> SimulationJob:
    status, interrupted = _finalize_loaded_status(str(data.get("status", INTERRUPTED_STATUS)))
    started = parse_iso_datetime(data.get("started_at"))
    if started is None:
        from ltspice_mcp.lib import now as _now

        started = _now()
    raw_file = Path(data["raw_file"]) if data.get("raw_file") else None
    log_file = Path(data["log_file"]) if data.get("log_file") else None
    job = SimulationJob(
        job_id=str(data["job_id"]),
        netlist=Path(str(data["netlist"])),
        simulator=str(data.get("simulator", "unknown")),
        status=status,  # type: ignore[arg-type]
        started_at=started,
        completed_at=parse_iso_datetime(data.get("completed_at")),
        raw_file=raw_file,
        log_file=log_file,
        error=("Server restarted while job was running" if interrupted else data.get("error")),
    )
    # Any loaded job is already terminal — pre-trigger the done event so
    # callers that await it don't block forever.
    if job.status in TERMINAL_STATUSES:
        job.done_event.set()
    return job


def _deserialize_sweep_config(data: dict | None) -> SweepConfig | None:
    if not data:
        return None
    dims = [
        SweepDimension(
            type=d.get("type", "component"),
            name=str(d.get("name", "")),
            start=float(d.get("start", 0.0)),
            stop=float(d.get("stop", 0.0)),
            step=d.get("step"),
            points=d.get("points"),
            scale=str(d.get("scale", "linear")),
        )
        for d in data.get("dimensions", [])
    ]
    return SweepConfig(netlist=Path(str(data.get("netlist", ""))), dimensions=dims)


def _deserialize_mc_config(data: dict | None) -> MonteCarloConfig | None:
    if not data:
        return None

    def _coerce_tol_map(raw: dict | None) -> dict[str, tuple[float, str]]:
        out: dict[str, tuple[float, str]] = {}
        for k, v in (raw or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[str(k)] = (float(v[0]), str(v[1]))
        return out

    return MonteCarloConfig(
        netlist=Path(str(data.get("netlist", ""))),
        type_tolerances=_coerce_tol_map(data.get("type_tolerances")),
        component_overrides=_coerce_tol_map(data.get("component_overrides")),
        num_runs=int(data.get("num_runs", 100)),
    )


def _deserialize_batch_job(data: dict) -> BatchJob:
    status, interrupted = _finalize_loaded_status(str(data.get("status", INTERRUPTED_STATUS)))
    started = parse_iso_datetime(data.get("started_at"))
    if started is None:
        from ltspice_mcp.lib import now as _now

        started = _now()

    run_results: dict[int, dict] = {}
    for key, res in (data.get("run_results") or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        run_results[idx] = {
            "raw_file": res.get("raw_file"),
            "log_file": res.get("log_file"),
            "params": dict(res.get("params") or {}),
        }

    bj = BatchJob(
        job_id=str(data["job_id"]),
        job_type=str(data.get("job_type", "sweep")),  # type: ignore[arg-type]
        netlist=Path(str(data["netlist"])),
        total_runs=int(data.get("total_runs", 0)),
        completed_runs=int(data.get("completed_runs", 0)),
        failed_runs=int(data.get("failed_runs", 0)),
        status=status,  # type: ignore[arg-type]
        started_at=started,
        completed_at=parse_iso_datetime(data.get("completed_at")),
        error=("Server restarted while job was running" if interrupted else data.get("error")),
        run_results=run_results,
        sweep_config=_deserialize_sweep_config(data.get("sweep_config")),
        mc_config=_deserialize_mc_config(data.get("mc_config")),
    )
    if bj.status in TERMINAL_STATUSES:
        bj.done_event.set()
    return bj


def load_jobs_for_circuit(
    circuit_path: Path,
) -> tuple[list[SimulationJob], list[BatchJob]]:
    """Scan a circuit's sidecar directory and return parsed jobs.

    Unparseable files are skipped with a warning rather than aborting the load.
    """
    target = sidecar_dir(circuit_path)
    sim_jobs: list[SimulationJob] = []
    batch_jobs: list[BatchJob] = []
    if not target.is_dir():
        return sim_jobs, batch_jobs

    for file_path in sorted(target.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping unreadable job file %s: %s", file_path, e)
            continue
        if not _accept_schema(data, file_path):
            continue
        kind = data.get("kind")
        try:
            if kind == "batch":
                batch_jobs.append(_deserialize_batch_job(data))
            else:
                sim_jobs.append(_deserialize_sim_job(data))
        except Exception as e:
            logger.warning("Skipping malformed job file %s: %s", file_path, e)
            continue

    return sim_jobs, batch_jobs


def summarize_circuit(circuit_path: Path) -> dict[str, Any]:
    """Return a lightweight summary of one circuit's persisted jobs.

    The sidecar dir is per-directory, so a single ``.ltspice-mcp/jobs/``
    folder holds records for every circuit in that directory. Filter to
    just the rows whose persisted ``netlist`` field matches ``circuit_path``
    — otherwise every circuit in the dir reports the directory's totals.
    """
    target = sidecar_dir(circuit_path)
    counts: dict[str, int] = {}
    interrupted_ids: list[str] = []
    total = 0
    try:
        match_path = str(circuit_path.resolve())
    except OSError:
        match_path = str(circuit_path)
    if target.is_dir():
        for file_path in target.glob("*.json"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not _accept_schema(data, file_path):
                continue
            record_netlist = str(data.get("netlist", ""))
            if record_netlist != match_path:
                continue
            # Running/queued in a persisted record means the prior server
            # died — treat as interrupted for summary purposes.
            status, _ = _finalize_loaded_status(str(data.get("status", "unknown")))
            counts[status] = counts.get(status, 0) + 1
            total += 1
            if status == INTERRUPTED_STATUS:
                jid = str(data.get("job_id", ""))
                if jid:
                    interrupted_ids.append(jid)
    return {
        "path": str(circuit_path),
        "exists": circuit_path.exists(),
        "total_jobs": total,
        "status_counts": counts,
        "interrupted_job_ids": interrupted_ids,
    }
