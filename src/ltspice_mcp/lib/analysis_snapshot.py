"""Versioned envelopes for durable attached-analysis results."""

from __future__ import annotations

from typing import Any

SNAPSHOT_KIND = "ltspice-mcp/attached-analysis-snapshot"
SNAPSHOT_VERSION = 1


def envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one neutral snapshot so it cannot be mistaken for a public result."""
    return {
        "kind": SNAPSHOT_KIND,
        "snapshot_version": SNAPSHOT_VERSION,
        **payload,
    }


def is_snapshot(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("kind") == SNAPSHOT_KIND
        and value.get("snapshot_version") == SNAPSHOT_VERSION
    )


def has_snapshot_kind(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") == SNAPSHOT_KIND
