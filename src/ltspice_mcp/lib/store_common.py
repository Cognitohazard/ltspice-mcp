"""Shared, dependency-light helpers for versioned JSON job stores."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ltspice_mcp.lib import atomic_write_json as _atomic_write_json

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def pid_of(data: Mapping[str, Any]) -> int | None:
    """Owning-server pid from a stored record, or None if absent or invalid."""
    pid = data.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def owner_alive(pid: int | None, *, own_is_alive: bool = False) -> bool:
    """Whether the record's owning server process is still running.

    ``own_is_alive`` decides how a record carrying this process's pid reads:
    registry loading treats it as a recycled pid, while disk-level summaries
    treat the common own-pid case as a genuinely running job.
    """
    if not pid:
        return False
    if pid == os.getpid():
        return own_is_alive
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def json_default(obj: Any) -> Any:
    """Serialize paths and datetimes shared by both job stores."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically and durably write a JSON document."""
    _atomic_write_json(path, data, default=json_default)


def schema_envelope(schema: str, version: int, **payload: Any) -> dict[str, Any]:
    """Build the common schema/version envelope used by persistent records."""
    return {"schema": schema, "schema_version": version, **payload}


def migrate_record(
    data: dict[str, Any],
    from_version: int,
    current_version: int,
    migrations: Mapping[int, Migration],
) -> dict[str, Any]:
    """Apply consecutive ``vN -> vN+1`` migrations to ``data``."""
    current = from_version
    while current < current_version:
        migrate_fn = migrations.get(current)
        if migrate_fn is None:
            raise ValueError(
                f"No migration path from schema_version {current} to {current_version}"
            )
        data = migrate_fn(data)
        current += 1
    data["schema_version"] = current_version
    return data


def accept_schema(
    data: dict[str, Any],
    source: Path,
    *,
    schema: str,
    current_version: int,
    supported_versions: frozenset[int],
    migrations: Mapping[int, Migration],
    logger: logging.Logger,
) -> bool:
    """Validate and, when supported, migrate a versioned JSON envelope."""
    found_schema = data.get("schema")
    if found_schema != schema:
        logger.warning(
            "Skipping job file %s: unexpected schema %r (expected %s)",
            source,
            found_schema,
            schema,
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
    if raw_version == current_version:
        return True
    if raw_version in supported_versions and raw_version < current_version:
        try:
            migrate_record(data, raw_version, current_version, migrations)
        except ValueError as exc:
            logger.warning("Skipping job file %s: %s", source, exc)
            return False
        return True

    logger.warning(
        "Skipping job file %s: unsupported schema_version %d (this build reads %s)",
        source,
        raw_version,
        sorted(supported_versions),
    )
    return False
