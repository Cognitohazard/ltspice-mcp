"""Per-circuit JSON persistence for job records.

Jobs are stored in ``{circuit_parent}/.ltspice-mcp/jobs/{job_id}.json`` so they
travel with the circuit they belong to. Loads are lazy — the server only reads a
circuit's sidecar directory the first time a tool touches that circuit.

This version writes experiment records only (through ``experiment_store``); the
simulation and batch records earlier releases wrote are read, never written.
They come back as :class:`LegacyJobRecord`: recognised, listed, and inert. A
directory full of them must not break the registry or the startup preload, and a
caller that asks about one must be told why it has no results rather than left
waiting on a job nothing will ever finish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    LegacyJobRecord,
)
from ltspice_mcp.lib.store_common import JOB_SCHEMA, accept_schema

logger = logging.getLogger(__name__)

SIDECAR_DIRNAME = ".ltspice-mcp"
JOBS_SUBDIR = "jobs"
SCHEMA = JOB_SCHEMA
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


def _migrate_v1_to_v2(data: dict) -> dict:
    """v1 -> v2: ``SweepDimension`` gained an optional ``values`` list and nullable
    ``start``/``stop`` (explicit discrete-value sweeps). No data transform is
    needed — those fields are no longer read at all — so this only re-stamps the
    version (done by ``_migrate``). Idempotent-safe: returns ``data`` unchanged."""
    return data


# Registered migration functions. Key N transforms v(N) into v(N+1).
_MIGRATIONS: dict[int, Any] = {1: _migrate_v1_to_v2}


def sidecar_dir(circuit_path: Path) -> Path:
    """Return the ``.ltspice-mcp/jobs`` directory next to a circuit file."""
    return circuit_path.parent / SIDECAR_DIRNAME / JOBS_SUBDIR


def _job_file(job_id: str, dir_: Path) -> Path:
    return dir_ / f"{job_id}.json"


def _effective_status(raw_status: str) -> str:
    """The status a pre-0.6 record reports here.

    Any live-looking status becomes ``interrupted``, whatever pid the file
    names: that pid belonged to a process running a release this one has no
    runner for, so nothing here can still be executing the job. Reporting it as
    running would leave a caller waiting on a job that will never report.
    """
    return INTERRUPTED_STATUS if raw_status in NON_TERMINAL_LIVE_STATUSES else raw_status


def _accept_schema(data: dict, source: Path) -> bool:
    """Verify a loaded record's schema is one we understand, migrating if needed.

    Modifies ``data`` in place when applying a migration so callers get the
    current-schema shape without special-casing versions. Returns False for
    unsupported versions or schemas (caller should skip that record).
    """
    return accept_schema(
        data,
        source,
        schema=SCHEMA,
        current_version=SCHEMA_VERSION,
        supported_versions=SUPPORTED_VERSIONS,
        migrations=_MIGRATIONS,
        logger=logger,
    )


def _read_legacy_record(data: dict) -> LegacyJobRecord:
    """Build the inert record a pre-0.6 sidecar becomes.

    Only the four fields a caller can still be told about survive: which job
    it was, which circuit it belonged to, what shape it claimed, and the
    status the earlier release last wrote. Everything the old readers needed
    (per-run results, sweep and Monte Carlo configs, artifact paths) is
    deliberately dropped — this version cannot act on any of it.
    """
    status = _effective_status(str(data.get("status", INTERRUPTED_STATUS)))
    return LegacyJobRecord(
        job_id=str(data.get("job_id", "")),
        netlist=Path(str(data.get("netlist", ""))),
        kind="batch" if data.get("kind") == "batch" else "sim",
        status=status,
    )


def _load_job_file(path: Path) -> LegacyJobRecord | ExperimentJob | None:
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
    if data.get("kind") == "experiment":
        try:
            from ltspice_mcp.lib import experiment_store

            working_dir = path.parent.parent.parent
            return experiment_store.load_job_from_path(
                path,
                working_dir,
                own_is_alive=True,
            )
        except Exception as e:
            logger.warning("Skipping malformed experiment job file %s: %s", path, e)
            return None
    if not _accept_schema(data, path):
        return None
    try:
        return _read_legacy_record(data)
    except Exception as e:
        logger.warning("Skipping malformed job file %s: %s", path, e)
        return None


def load_jobs_for_circuit(circuit_path: Path) -> list[LegacyJobRecord]:
    """Every pre-0.6 record in a circuit's sidecar directory.

    Unparseable files are skipped with a warning rather than aborting the load.
    Experiment records are this session's own and are loaded by the experiment
    store, not here.
    """
    target = sidecar_dir(circuit_path)
    if not target.is_dir():
        return []
    records: list[LegacyJobRecord] = []
    for file_path in sorted(target.glob("*.json")):
        job = _load_job_file(file_path)
        if isinstance(job, LegacyJobRecord):
            records.append(job)
    return records


def load_job(job_id: str, netlist: Path) -> LegacyJobRecord | ExperimentJob | None:
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
    """Return a lightweight summary of one circuit's persisted job records.

    The sidecar dir is per-directory, so a single ``.ltspice-mcp/jobs/``
    folder holds records for every circuit in that directory. Filter to
    just the rows whose persisted ``netlist`` field matches ``circuit_path``
    — otherwise every circuit in the dir reports the directory's totals.

    A batch record (``kind="batch"``) shows up as ONE entry under ``total_jobs``
    but has ``total_runs`` underlying simulation iterations. Both numbers
    are surfaced separately so a 100-run Monte Carlo isn't mistaken for
    "circuit ran once".
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
            status = _effective_status(str(data.get("status", "unknown")))
            counts[status] = counts.get(status, 0) + 1
            total += 1
            if data.get("kind") == "batch":
                runs = data.get("total_runs")
                total_runs += runs if isinstance(runs, int) and runs > 0 else 1
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
