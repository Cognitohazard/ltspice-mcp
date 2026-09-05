"""Immutable persistence for continuable ``analyze_results`` calls."""

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
from ltspice_mcp.lib.cursor_codec import CursorError
from ltspice_mcp.lib.cursor_codec import canonical_hash as canonical_hash
from ltspice_mcp.lib.cursor_codec import canonical_json as canonical_json
from ltspice_mcp.lib.cursor_codec import decode_cursor as _decode_body_cursor
from ltspice_mcp.lib.cursor_codec import encode_cursor as _encode_body_cursor
from ltspice_mcp.lib.deck_staging import sha256_file as sha256_file  # re-export
from ltspice_mcp.lib.store import KIND_RESULT_SET, Store, accept, atomic_write_json, envelope

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


def composite_digest(raw_sha256: str, log_sha256: str | None, log_present: bool) -> str:
    """Canonical digest of the raw plus the log digest-or-recorded-absence."""
    return canonical_hash(
        {
            "raw_sha256": raw_sha256,
            "log": (
                {"present": True, "sha256": log_sha256} if log_present else {"present": False}
            ),
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
    """Persist a new immutable result set after opportunistic cleanup."""
    cleanup(working_dir)
    created = now()
    result_set_id = f"rs_{secrets.token_hex(16)}"
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
    item = replace(item, snapshot_hash=canonical_hash(item.snapshot()))
    store = Store(working_dir)
    store.ensure_root()
    atomic_write_json(store.result_set(result_set_id), item.to_dict())
    return item


def _decode(data: dict[str, Any], path: Path) -> ResultSet:
    if not accept(data, path, kind=KIND_RESULT_SET, log=logger):
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


def load(result_set_id: str, working_dir: Path) -> ResultSet:
    """Load one immutable set, enforcing TTL and source-job retention."""
    path = result_path(result_set_id, working_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ResultError(
            f"Result set {result_set_id!r} is missing or expired; submit a fresh "
            "analyze_results request."
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Could not read result set {result_set_id!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultError(f"Result set {result_set_id!r} is malformed")
    item = _decode(data, path)
    # Job-backed sets follow their jobs and are not subject to the raw-only TTL.
    expires_at = parse_iso_datetime(item.expires_at)
    if expires_at is None:
        raise ResultError(f"Result set {result_set_id!r} has an invalid expiry timestamp")
    if not item.source_jobs and expires_at <= now():
        with contextlib.suppress(OSError):
            path.unlink()
        _delete_artifacts(working_dir, result_set_id)
        raise ResultError(
            f"Result set {result_set_id!r} expired; submit a fresh analyze_results request."
        )
    missing = _job_record_missing(item)
    if missing is not None:
        with contextlib.suppress(OSError):
            path.unlink()
        _delete_artifacts(working_dir, result_set_id)
        raise ResultError(
            f"Result set {result_set_id!r} is invalid because source job {missing!r} "
            "was deleted; submit a fresh analyze_results request."
        )
    if item.work_hash != canonical_hash(item.work):
        raise ResultError(f"Result set {result_set_id!r} failed its integrity check")
    if item.snapshot_hash != canonical_hash(item.snapshot()):
        raise ResultError(f"Result set {result_set_id!r} failed its integrity check")
    return item


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


def encode_cursor(
    item: ResultSet,
    position: int,
    *,
    intra_item: int = 0,
    missing_offset: int = 0,
    view_fields: list[str] | None,
) -> str:
    """Encode a resume point: work position, per-run offset, coverage offset.

    ``missing_offset`` pages the coverage view (``missing_cases``), which lives
    in the immutable inputs rather than the work list — a cursor that carries it
    with ``position == len(work)`` pages that view without redoing any work.
    Every cursor names the row view it was paged under, ``None`` being the lean
    view; a cursor carrying no view at all does not decode.
    """
    body: dict[str, Any] = {
        "result_set_id": item.result_set_id,
        "position": position,
        "intra_item": intra_item,
        "missing_offset": missing_offset,
        "work_hash": item.work_hash,
        "view": {"fields": view_fields},
    }
    return _encode_body_cursor(body)


def _decode_cursor_body(cursor: str) -> dict[str, Any]:
    try:
        return _decode_body_cursor(cursor)
    except CursorError as exc:
        raise ResultError(f"Invalid analyze_results cursor: {exc}") from None


def cursor_result_set_id(cursor: str) -> str:
    """Recover the immutable set id carried by a callable page cursor."""
    result_set_id = str(_decode_cursor_body(cursor).get("result_set_id", ""))
    if not result_set_id:
        raise ResultError("Invalid analyze_results cursor: result_set_id is missing")
    return result_set_id


def cursor_view(cursor: str) -> list[str] | None:
    """Return the fields projection the cursor was paged under (``None`` = lean)."""
    body = _decode_cursor_body(cursor)
    if "view" not in body:
        raise ResultError("Invalid analyze_results cursor: it names no row view")
    view = body["view"]
    if not isinstance(view, dict) or set(view) != {"fields"}:
        raise ResultError("Invalid analyze_results cursor: malformed render view")
    fields = view["fields"]
    if fields is not None and (
        not isinstance(fields, list) or not all(isinstance(field, str) for field in fields)
    ):
        raise ResultError("Invalid analyze_results cursor: malformed fields render view")
    return fields


def reencode_cursor(
    cursor: str,
    *,
    view_fields: list[str] | None,
    position: int | None = None,
    intra_item: int | None = None,
    missing_offset: int | None = None,
) -> str:
    """Re-render one immutable resume point with a selected view or offset."""
    body = _decode_cursor_body(cursor)
    body["view"] = {"fields": view_fields}
    if position is not None:
        body["position"] = position
    if intra_item is not None:
        body["intra_item"] = intra_item
    if missing_offset is not None:
        body["missing_offset"] = missing_offset
    return _encode_body_cursor(body)


def decode_cursor(cursor: str, item: ResultSet) -> tuple[int, int, int]:
    """Decode and bind a cursor to one immutable result set and work list.

    Returns ``(position, intra_item, missing_offset)``.
    """
    try:
        body = _decode_cursor_body(cursor)
        if body["result_set_id"] != item.result_set_id or body["work_hash"] != item.work_hash:
            raise ValueError("cursor belongs to a different result set")
        position = int(body["position"])
        intra_item = int(body.get("intra_item", 0))
        missing_offset = int(body.get("missing_offset", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultError(f"Invalid analyze_results cursor: {exc}") from None
    if position < 0 or position > len(item.work) or intra_item < 0 or missing_offset < 0:
        raise ResultError("Invalid analyze_results cursor position")
    return position, intra_item, missing_offset


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
