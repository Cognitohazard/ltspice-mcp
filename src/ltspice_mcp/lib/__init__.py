"""Library utilities for ltspice-mcp."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_EST = timezone(timedelta(hours=-5), name="EST")

# Recognised netlist / schematic file extensions. Shared by the recent-
# circuits tracker, sidecar loaders, and the netlist resource listing.
CIRCUIT_EXTENSIONS: frozenset[str] = frozenset(
    {".asc", ".net", ".sp", ".cir", ".spice"}
)


def now() -> datetime:
    """Return the current time in US Eastern (EST, UTC-5)."""
    return datetime.now(tz=_EST)


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp defensively; returns None on failure or non-string input."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    indent: int = 2,
) -> None:
    """Write ``data`` as JSON to ``path`` atomically.

    Creates parent directories, writes to a tempfile in the same directory,
    then renames into place. ``os.replace`` is atomic on POSIX and overwrites
    on Windows, so concurrent readers never see a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=default)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
