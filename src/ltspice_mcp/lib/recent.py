"""Global index of recently-touched circuit files.

The index lives in a user-global state directory so a new session can
surface prior work no matter which project it was started in. Resolution
order (first match wins):

1. ``$LTSPICE_MCP_HOME/recent.json`` (tests and explicit overrides).
2. ``$XDG_STATE_HOME/ltspice-mcp/recent.json`` (XDG Base Directory).
3. ``~/.local/state/ltspice-mcp/recent.json`` (XDG default base).

Writes are serialised across processes via ``file_lock`` — parallel MCP
sessions sharing a machine won't lose entries to read-modify-write races.
The file contains only absolute paths and ISO timestamps.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

from ltspice_mcp.lib import CIRCUIT_EXTENSIONS, atomic_write_json, now
from ltspice_mcp.lib.filelock import file_lock

logger = logging.getLogger(__name__)

DEFAULT_CAP = 20
INDEX_FILENAME = "recent.json"

SCHEMA = "ltspice-mcp/recent"
SCHEMA_VERSION = 1
# Versions this build can READ. A file must carry a recognised
# ``schema_version``; anything else is ignored.
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})


def index_path() -> Path:
    """Resolve the global recent-circuits file path.

    See module docstring for the full resolution order. In short: explicit
    ``$LTSPICE_MCP_HOME`` always wins; otherwise the XDG state directory
    (``$XDG_STATE_HOME`` or the ``~/.local/state`` default) is used.
    """
    override = os.getenv("LTSPICE_MCP_HOME")
    if override:
        return Path(override) / INDEX_FILENAME
    xdg = os.getenv("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "ltspice-mcp" / INDEX_FILENAME


def is_circuit_file(path: Path) -> bool:
    """True if ``path`` has a recognised netlist/schematic extension."""
    return path.suffix.lower() in CIRCUIT_EXTENSIONS


def _read_index(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable recent index %s: %s", path, e)
        return []
    if not isinstance(data, dict):
        return []
    # Schema check: require a recognised ``schema_version``. Reject
    # versionless files and anything from a newer build so we don't
    # silently drop unknown fields on the next write.
    raw_version = data.get("schema_version")
    if not isinstance(raw_version, int) or raw_version not in SUPPORTED_VERSIONS:
        logger.warning(
            "Ignoring recent index %s: unsupported schema_version %r",
            path,
            raw_version,
        )
        return []
    entries = data.get("circuits")
    if not isinstance(entries, list):
        return []
    out: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        p = entry.get("path")
        t = entry.get("last_touched")
        if isinstance(p, str) and p:
            out.append({"path": p, "last_touched": t if isinstance(t, str) else None})
    return out


def _index_payload(entries: list[dict]) -> dict:
    """Build the full on-disk payload: schema header + circuit entries."""
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "circuits": entries,
    }


def touch(circuit_path: Path, cap: int = DEFAULT_CAP) -> None:
    """Record a circuit as recently touched, bumping it to the top.

    Silently skips non-circuit files so callers can invoke this
    unconditionally after a tool dispatch. The read-modify-write is
    serialised across processes via a sibling lock file.
    """
    if not is_circuit_file(circuit_path):
        return
    try:
        resolved = str(circuit_path.resolve())
    except OSError:
        return
    path = index_path()
    try:
        with file_lock(path):
            entries = _read_index(path)
            entries = [e for e in entries if e.get("path") != resolved]
            entries.insert(0, {"path": resolved, "last_touched": now().isoformat()})
            if len(entries) > cap:
                entries = entries[:cap]
            atomic_write_json(path, _index_payload(entries))
    except (OSError, TimeoutError) as e:
        logger.warning("Failed to persist recent-circuits index %s: %s", path, e)


def load(*, prune_missing: bool = False) -> list[dict]:
    """Return the recent-circuits list, newest first.

    When ``prune_missing`` is set, entries whose files no longer exist are
    dropped from both the returned list and the on-disk index.
    """
    path = index_path()
    if not prune_missing:
        return _read_index(path)
    try:
        with file_lock(path):
            entries = _read_index(path)
            kept: list[dict] = []
            dropped = False
            for entry in entries:
                p = entry.get("path")
                if isinstance(p, str) and Path(p).exists():
                    kept.append(entry)
                else:
                    dropped = True
            if dropped:
                with contextlib.suppress(OSError):
                    atomic_write_json(path, _index_payload(kept))
            return kept
    except (OSError, TimeoutError) as e:
        logger.warning("Failed to prune recent-circuits index %s: %s", path, e)
        return _read_index(path)
