"""Per-circuit JSON persistence for simulation and batch jobs.

Jobs are stored in ``{circuit_parent}/.ltspice-mcp/jobs/{job_id}.json`` so they
travel with the circuit they belong to. Writes are atomic (tempfile + rename).
Loads are lazy — the server only reads a circuit's sidecar directory the first
time a tool touches that circuit in a session.

Jobs whose server process died while they were running come back as
``interrupted``; a record whose owning process is still alive — a parallel
server session's live job — keeps its status as written (see
``_finalize_loaded_status``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

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
# v2 (2026-05-30): SweepDimension gained an optional ``values`` list and nullable
# ``start``/``stop`` for explicit discrete-value sweeps. The shape change is why
# the version bumped — so a v1-only reader rejects v2 records via _accept_schema
# instead of crashing on ``float(None)`` for a null ``start``.
SCHEMA_VERSION = 2
# Versions this build can READ after applying ``_MIGRATIONS``. Always
# includes the current version; older versions are added once their
# migration function lands in ``_MIGRATIONS``.
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2})
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


def _migrate_v1_to_v2(data: dict) -> dict:
    """v1 -> v2: ``SweepDimension`` gained an optional ``values`` list and nullable
    ``start``/``stop`` (explicit discrete-value sweeps). No data transform is
    needed — the v2 reader treats a missing ``values`` as ``None`` and reads v1's
    always-present ``start``/``stop`` unchanged — so this only re-stamps the
    version (done by ``_migrate``). Idempotent-safe: returns ``data`` unchanged."""
    return data


# Registered migration functions. Key N transforms v(N) into v(N+1).
# Keep each function focused and reversible where possible.
_MIGRATIONS: dict[int, Any] = {1: _migrate_v1_to_v2}


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
        # Additive within schema v2: older records lack it; readers treat a
        # missing pid as a dead owner (the pre-pid behavior).
        "pid": job.owner_pid,
        "netlist": str(job.netlist),
        "simulator": job.simulator,
        "status": job.status,
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "raw_file": str(job.raw_file) if job.raw_file else None,
        "log_file": str(job.log_file) if job.log_file else None,
        "error": job.error,
        # Additive within schema v2: older records lack these; readers treat
        # a missing key as "no alias requested" (the pre-alias behavior).
        "output_basename": job.output_basename,
        "output_alias_raw": str(job.output_alias_raw) if job.output_alias_raw else None,
        "output_alias_log": str(job.output_alias_log) if job.output_alias_log else None,
        "output_alias_note": job.output_alias_note,
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
        "pid": job.owner_pid,
        "job_type": job.job_type,
        "netlist": str(job.netlist),
        "simulator": job.simulator,
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


def _pid_of(data: dict) -> int | None:
    """Owning-server pid from a job record, or None if absent/invalid."""
    pid = data.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _owner_alive(pid: int | None, *, own_is_alive: bool = False) -> bool:
    """Whether the record's owning server process is still running.

    ``own_is_alive`` decides how a record carrying OUR pid reads, because
    the right answer depends on the caller. Registry loading passes False:
    a record we're loading isn't in our registry, so a matching pid can only
    be a recycled one — dead owner. Disk-level summaries pass True: there
    the overwhelmingly common own-pid case is this server's genuinely
    running job (which registry loading never sees — it dedups against
    in-memory jobs first).

    Liveness only — a pid recycled by an unrelated process also reads as
    alive until it exits; compare process create_time against the job's
    started_at if that ever matters in practice.
    """
    if not pid:
        return False
    if pid == os.getpid():
        return own_is_alive
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _finalize_loaded_status(
    raw_status: str, owner_pid: int | None = None, *, own_is_alive: bool = False
) -> tuple[str, bool]:
    """Translate a loaded status.

    Returns (effective_status, was_interrupted). Running/queued jobs whose
    owning process is gone come back as ``interrupted``; if the owner is
    still alive (a parallel server session's live job), the status stands
    as written. ``own_is_alive`` is forwarded to ``_owner_alive`` — see its
    docstring for which callers pass True.
    """
    if raw_status in NON_TERMINAL_LIVE_STATUSES and not _owner_alive(
        owner_pid, own_is_alive=own_is_alive
    ):
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
    pid = _pid_of(data)
    status, interrupted = _finalize_loaded_status(str(data.get("status", INTERRUPTED_STATUS)), pid)
    started = parse_iso_datetime(data.get("started_at"))
    if started is None:
        from ltspice_mcp.lib import now as _now

        started = _now()
    raw_file = Path(data["raw_file"]) if data.get("raw_file") else None
    log_file = Path(data["log_file"]) if data.get("log_file") else None
    alias_raw = Path(data["output_alias_raw"]) if data.get("output_alias_raw") else None
    alias_log = Path(data["output_alias_log"]) if data.get("output_alias_log") else None
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
        # The record's pid, NOT ours: a loaded job belongs to whichever
        # process persisted it (0 when the record predates the pid field).
        owner_pid=pid or 0,
        output_basename=data.get("output_basename"),
        output_alias_raw=alias_raw,
        output_alias_log=alias_log,
        output_alias_note=data.get("output_alias_note"),
    )
    # A loaded terminal job's work is over — pre-trigger the done event so
    # callers that await it don't block forever. (A parallel session's live
    # job stays unset; its completion is signalled only in the owner.)
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
            start=None if d.get("start") is None else float(d["start"]),
            stop=None if d.get("stop") is None else float(d["stop"]),
            step=d.get("step"),
            points=d.get("points"),
            scale=str(d.get("scale", "linear")),
            values=([float(v) for v in d["values"]] if d.get("values") is not None else None),
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
    pid = _pid_of(data)
    status, interrupted = _finalize_loaded_status(str(data.get("status", INTERRUPTED_STATUS)), pid)
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
        simulator=str(data.get("simulator", "")),
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
        owner_pid=pid or 0,
    )
    if bj.status in TERMINAL_STATUSES:
        bj.done_event.set()
    return bj


def _load_job_file(path: Path) -> SimulationJob | BatchJob | None:
    """Read + schema-check + deserialize one sidecar record, or None.

    Unreadable, unsupported-schema, and malformed files log a warning and
    return None — a bad record never aborts a directory load.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Skipping unreadable job file %s: %s", path, e)
        return None
    if not _accept_schema(data, path):
        return None
    try:
        if data.get("kind") == "batch":
            return _deserialize_batch_job(data)
        return _deserialize_sim_job(data)
    except Exception as e:
        logger.warning("Skipping malformed job file %s: %s", path, e)
        return None


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
        job = _load_job_file(file_path)
        if isinstance(job, BatchJob):
            batch_jobs.append(job)
        elif isinstance(job, SimulationJob):
            sim_jobs.append(job)

    return sim_jobs, batch_jobs


def load_job(job_id: str, netlist: Path) -> SimulationJob | BatchJob | None:
    """Load one job record by id from its circuit's sidecar, or None.

    Used to refresh this session's view of a job owned by a parallel server
    process — the owner keeps persisting status changes the in-memory
    registry would otherwise never see. A missing file (e.g. the owner
    evicted the job) is a silent None, not a warning.
    """
    path = _job_file(job_id, sidecar_dir(netlist))
    if not path.is_file():
        return None
    return _load_job_file(path)


def summarize_circuit(circuit_path: Path) -> dict[str, Any]:
    """Return a lightweight summary of one circuit's persisted jobs.

    The sidecar dir is per-directory, so a single ``.ltspice-mcp/jobs/``
    folder holds records for every circuit in that directory. Filter to
    just the rows whose persisted ``netlist`` field matches ``circuit_path``
    — otherwise every circuit in the dir reports the directory's totals.

    A batch job (``kind="batch"``) shows up as ONE entry under ``total_jobs``
    but has ``total_runs`` underlying simulation iterations. Both numbers
    are surfaced separately so a 100-run MC isn't mistaken for "circuit
    ran once".
    """
    target = sidecar_dir(circuit_path)
    counts: dict[str, int] = {}
    interrupted_ids: list[str] = []
    total = 0
    total_runs = 0
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
            # Running/queued with a dead owner means that server died —
            # summarize as interrupted; with a live owner the status stands.
            # own_is_alive: our own pid on a running record here is almost
            # always THIS server's live job (not a recycled pid), so the
            # summary reports it as running.
            status, _ = _finalize_loaded_status(
                str(data.get("status", "unknown")), _pid_of(data), own_is_alive=True
            )
            counts[status] = counts.get(status, 0) + 1
            total += 1
            if data.get("kind") == "batch":
                runs = data.get("total_runs")
                if isinstance(runs, int) and runs > 0:
                    total_runs += runs
                else:
                    total_runs += 1
            else:
                total_runs += 1
            if status == INTERRUPTED_STATUS:
                jid = str(data.get("job_id", ""))
                if jid:
                    interrupted_ids.append(jid)
    return {
        "path": str(circuit_path),
        "exists": circuit_path.exists(),
        "total_jobs": total,
        "total_runs": total_runs,
        "status_counts": counts,
        "interrupted_job_ids": interrupted_ids,
    }
