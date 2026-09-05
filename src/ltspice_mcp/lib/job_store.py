"""Reading the job sidecars a release before 0.6 left beside a circuit.

Those releases wrote ``{circuit_parent}/.ltspice-mcp/jobs/{job_id}.json`` so a
job travelled with its circuit. Nothing writes there any more — 0.6 keeps its
records in the working-directory store (``lib/store.py``), which is a different
directory even when the circuit sits in the working directory itself, so the two
formats never share a folder. These records are read, never written, and come
back as :class:`LegacyJobRecord`: recognised, listed, and inert. A directory
full of them must not break the registry or the startup preload, and a caller
that asks about one must be told why it has no results rather than left waiting
on a job nothing will ever finish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    LegacyJobRecord,
)
from ltspice_mcp.lib.store import SIDECAR_DIRNAME, Store

logger = logging.getLogger(__name__)

#: The schema name those releases stamped on a job sidecar.
SCHEMA = "ltspice-mcp/job"
#: The versions they wrote. Closed: no release will ever add a third, and this
#: build reads only the four fields a caller can still be told about, which both
#: versions carry — so there is nothing to migrate between them.
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2})
INTERRUPTED_STATUS = "interrupted"

__all__ = [
    "SCHEMA",
    "SIDECAR_DIRNAME",
    "SUPPORTED_VERSIONS",
    "load_job",
    "load_jobs_for_circuit",
    "sidecar_dir",
    "summarize_circuit",
]


def sidecar_dir(circuit_path: Path) -> Path:
    """Return the ``.ltspice-mcp/jobs`` directory next to a circuit file."""
    return Store.legacy_jobs_dir(circuit_path)


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


def _accept_schema(data: Any, source: Path) -> bool:
    """Whether a file in the legacy sidecar directory is one of these records.

    Returns False for anything else, with a warning: this directory was only
    ever written by those releases, so a file in it carrying some other schema
    is genuinely unexpected.
    """
    if not isinstance(data, dict):
        logger.warning("Skipping job file %s: not a JSON object", source)
        return False
    if data.get("schema") != SCHEMA:
        logger.warning(
            "Skipping job file %s: unexpected schema %r (expected %s)",
            source,
            data.get("schema"),
            SCHEMA,
        )
        return False
    version = data.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        logger.warning(
            "Skipping job file %s: unsupported schema_version %r (this build reads %s)",
            source,
            version,
            sorted(SUPPORTED_VERSIONS),
        )
        return False
    return True


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


def _load_job_file(path: Path) -> LegacyJobRecord | None:
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
        return _read_legacy_record(data)
    except Exception as e:
        logger.warning("Skipping malformed job file %s: %s", path, e)
        return None


def load_jobs_for_circuit(circuit_path: Path) -> list[LegacyJobRecord]:
    """Every pre-0.6 record in a circuit's sidecar directory.

    Unparseable files are skipped with a warning rather than aborting the load.
    """
    target = sidecar_dir(circuit_path)
    if not target.is_dir():
        return []
    records: list[LegacyJobRecord] = []
    for file_path in sorted(target.glob("*.json")):
        job = _load_job_file(file_path)
        if job is not None:
            records.append(job)
    return records


def load_job(job_id: str, netlist: Path) -> LegacyJobRecord | None:
    """Load one pre-0.6 record by id from its circuit's sidecar, or None.

    A missing file is a silent None, not a warning.
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
