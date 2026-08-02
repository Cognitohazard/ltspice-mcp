"""Versioned envelopes for durable attached-analysis results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.store_common import accept_schema, schema_envelope

logger = logging.getLogger(__name__)

SCHEMA = "ltspice-mcp/attached-analysis-snapshot"
SNAPSHOT_VERSION = 2
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, SNAPSHOT_VERSION})
_SOURCE = Path("<attached-analysis-snapshot>")


def _plain_presence(node: Any) -> dict[str, Any]:
    """Migrate the v1 flag/children tree to its record-shaped union."""
    if not isinstance(node, dict):
        return {}
    children = node.get("children")
    if not isinstance(children, dict):
        return {}
    return {str(key): _plain_presence(child) for key, child in children.items()}


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the duplicate answer assembly and normalize projection presence."""
    answer_top = data.pop("answer_top", None)
    natural_cursor = data.pop("answer_coverage_cursor_base", None)
    if not isinstance(natural_cursor, str) and isinstance(answer_top, dict):
        natural_cursor = answer_top.get("cursor")
    data["natural_cursor_base"] = natural_cursor
    data["natural_has_next"] = bool(
        isinstance(answer_top, dict) and isinstance(answer_top.get("next"), dict)
    )
    data["natural_deferred"] = bool(
        isinstance(answer_top, dict)
        and "artifact item was deferred intact" in str(answer_top.get("hint", ""))
    )
    results = data.get("results")
    if isinstance(results, dict):
        for block in results.values():
            if not isinstance(block, dict):
                continue
            block.pop("answer_facts", None)
            block["projection_presence"] = _plain_presence(block.get("projection_presence"))
    return data


_MIGRATIONS = {1: _migrate_v1_to_v2}


def envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one neutral snapshot so it cannot be mistaken for a public result."""
    return schema_envelope(SCHEMA, SNAPSHOT_VERSION, **payload)


def classify(value: Any) -> Literal["snapshot", "unsupported", "legacy"]:
    """Classify and migrate a stored attached-analysis value in place."""
    if not isinstance(value, dict):
        return "legacy"

    # Snapshot v1 predated the shared store envelope. Normalize its vocabulary
    # before handing version acceptance and migration to the common mechanism.
    if value.get("kind") == SCHEMA:
        value["schema"] = value.pop("kind")
        value["schema_version"] = value.pop("snapshot_version", None)
    elif value.get("schema") != SCHEMA:
        return "legacy"

    accepted = accept_schema(
        value,
        _SOURCE,
        schema=SCHEMA,
        current_version=SNAPSHOT_VERSION,
        supported_versions=SUPPORTED_VERSIONS,
        migrations=_MIGRATIONS,
        logger=logger,
    )
    return "snapshot" if accepted else "unsupported"
