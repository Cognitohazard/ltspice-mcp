"""The record of one ``analyze_results`` request, kept for its artifacts.

A result set records the request and the worker-captured source digests when
an evaluation starts. Nothing reads it to read on in a reply: a cut reply is a
stored answer (``lib/presenter.py``). It names the folder the request's CSV
and plot artifacts are written to (:func:`artifact_paths`), and its sweep
removes them with it: a set read from jobs when one of its jobs is deleted,
one read only from files at ``[analysis] result_set_ttl_hours``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import shutil
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import now, parse_iso_datetime
from ltspice_mcp.lib.cursor_codec import canonical_hash as canonical_hash
from ltspice_mcp.lib.cursor_codec import canonical_json as canonical_json
from ltspice_mcp.lib.deck_staging import sha256_file as sha256_file  # re-export
from ltspice_mcp.lib.store import (
    KIND_RESULT_SET,
    STORE_VERSION,
    Store,
    accept,
    atomic_write_json,
    envelope,
)

logger = logging.getLogger(__name__)


def result_root(working_dir: Path) -> Path:
    return Store(working_dir).results_dir


def result_path(result_set_id: str, working_dir: Path) -> Path:
    suffix = result_set_id[3:]
    if (
        not result_set_id.startswith("rs_")
        or len(suffix) != 32
        or any(char not in "0123456789abcdef" for char in suffix)
    ):
        raise ResultError(f"Invalid result_set_id: {result_set_id!r}")
    try:
        return Store(working_dir).result_set(result_set_id)
    except ValueError as exc:
        raise ResultError(f"Invalid result_set_id: {result_set_id!r}") from exc


def composite_digest(
    *,
    raw_sha256: str | None,
    log_sha256: str | None,
    console_sha256: str | None,
) -> str:
    """Bind RAW, log and console bytes, distinguishing empty files from absence."""
    return canonical_hash(
        {
            role: {"present": digest is not None, "sha256": digest}
            for role, digest in (
                ("raw", raw_sha256),
                ("log", log_sha256),
                ("console", console_sha256),
            )
        }
    )


@dataclass(frozen=True)
class ResultSet:
    result_set_id: str
    created_at: str
    expires_at: str
    inputs: dict[str, Any]
    work: list[dict[str, Any]]
    source_manifests: list[dict[str, Any]]
    source_jobs: dict[str, str | None]
    work_hash: str
    snapshot_hash: str

    def snapshot(self) -> dict[str, Any]:
        """The identity-bearing fields the ``snapshot_hash`` covers.

        One definition, shared by ``create`` (to compute the hash) and ``load``
        (to re-verify it), so the two can't drift into hashing different keys.
        """
        return {
            "result_set_id": self.result_set_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "inputs": self.inputs,
            "work": self.work,
            "source_manifests": self.source_manifests,
            "source_jobs": self.source_jobs,
        }

    def to_dict(self) -> dict[str, Any]:
        return envelope(
            KIND_RESULT_SET,
            **self.snapshot(),
            work_hash=self.work_hash,
            snapshot_hash=self.snapshot_hash,
        )


def create(
    *,
    working_dir: Path,
    inputs: dict[str, Any],
    work: list[dict[str, Any]],
    source_manifests: list[dict[str, Any]],
    source_jobs: dict[str, str | None],
    ttl_hours: float,
) -> ResultSet:
    """Persist a result set after opportunistic cleanup.

    Its id is a digest of what it records: the request, the sources' digests
    and the jobs it reads, so the same request over the same files is one set
    (written again, with a fresh expiry), its artifacts keep their names, and
    a reply built from it keeps its bytes (design principle "Repeatable
    requests")."""
    cleanup(working_dir)
    created = now()
    result_set_id = "rs_" + canonical_hash(
        {
            "inputs": inputs,
            "work": work,
            "source_manifests": source_manifests,
            "source_jobs": source_jobs,
        }
    )[:32]
    created_at = created.isoformat()
    expires_at = (created + timedelta(hours=ttl_hours)).isoformat()
    item = ResultSet(
        result_set_id=result_set_id,
        created_at=created_at,
        expires_at=expires_at,
        inputs=inputs,
        work=work,
        source_manifests=source_manifests,
        source_jobs=source_jobs,
        work_hash=canonical_hash(work),
        snapshot_hash="",
    )
    store = Store(working_dir)
    store.ensure_root()
    return _persist(item, store)


def _persist(item: ResultSet, store: Store) -> ResultSet:
    """Stamp a new set's snapshot hash and write its immutable record."""
    item = replace(item, snapshot_hash=canonical_hash(item.snapshot()))
    atomic_write_json(store.result_set(item.result_set_id), item.to_dict())
    return item


def _decode(data: dict[str, Any], path: Path) -> ResultSet:
    if data.get("store_version") != STORE_VERSION or not accept(
        data, path, kind=KIND_RESULT_SET, log=logger
    ):
        raise ResultError(f"Result set {path.stem!r} uses an unsupported storage schema")
    try:
        return ResultSet(
            result_set_id=str(data["result_set_id"]),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
            inputs=dict(data["inputs"]),
            work=[dict(item) for item in data["work"]],
            source_manifests=[dict(item) for item in data["source_manifests"]],
            source_jobs={
                str(job_id): (str(record) if record is not None else None)
                for job_id, record in dict(data["source_jobs"]).items()
            },
            work_hash=str(data["work_hash"]),
            snapshot_hash=str(data["snapshot_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultError(f"Result set {path.stem!r} is malformed: {exc}") from exc


def _job_record_missing(item: ResultSet) -> str | None:
    for job_id, raw_path in item.source_jobs.items():
        if raw_path is not None and not Path(raw_path).is_file():
            return job_id
    return None


def _delete_artifacts(working_dir: Path, result_set_id: str) -> None:
    with contextlib.suppress(OSError, ValueError):
        shutil.rmtree(Store(working_dir).result_artifacts(result_set_id))


# The stored answers beside these sets (lib/presenter.py) are swept by
# presenter.cleanup: they depend on no job, so they expire by time alone.
def cleanup(working_dir: Path) -> int:
    """Delete expired raw-only sets and sets whose persisted source job vanished."""
    root = result_root(working_dir)
    removed = 0
    try:
        paths = list(root.glob("rs_*.json"))
    except OSError:
        return 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            item = _decode(data, path)
            expires_at = parse_iso_datetime(item.expires_at)
            expired = not item.source_jobs and expires_at is not None and expires_at <= now()
            invalid = _job_record_missing(item) is not None
            if expired or invalid:
                path.unlink()
                _delete_artifacts(working_dir, item.result_set_id)
                removed += 1
        except (OSError, ValueError, ResultError, json.JSONDecodeError):
            continue
    return removed


def invalidate_for_job(working_dir: Path, job_id: str) -> int:
    """Delete every result set that requires ``job_id`` as a source."""
    root = result_root(working_dir)
    removed = 0
    try:
        paths = list(root.glob("rs_*.json"))
    except OSError:
        return 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            source_jobs = data.get("source_jobs", {})
            if isinstance(source_jobs, dict) and job_id in source_jobs:
                path.unlink()
                result_set_id = str(data.get("result_set_id", path.stem))
                _delete_artifacts(working_dir, result_set_id)
                removed += 1
        except (OSError, json.JSONDecodeError):
            continue
    return removed


def artifact_paths(
    item: ResultSet,
    *,
    recipe_key: str,
    source_digest: str,
    recipe_hash: str,
    suffix: str,
) -> tuple[Path, Path]:
    """Return a unique pending path and the deterministic artifact target."""
    identity = canonical_hash(
        {
            "result_set_id": item.result_set_id,
            "recipe_key": recipe_key,
            "source_digest": source_digest,
            "recipe_hash": recipe_hash,
        }
    )
    root = Store(Path(item.inputs["working_dir"])).result_artifacts(item.result_set_id)
    final = root / f"{identity}.{suffix.lstrip('.')}"
    pending = root / f".{identity}.{os.getpid()}.{secrets.token_hex(8)}.pending"
    return pending, final
