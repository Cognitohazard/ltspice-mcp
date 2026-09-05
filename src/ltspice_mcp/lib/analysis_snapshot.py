"""The envelope around a durable attached-analysis result.

An experiment's attached analysis is computed once and stored inside the job
record, where it sits next to values a caller reads directly. The envelope is
what keeps the two apart: a stored analysis carrying it is this build's own
neutral snapshot, and anything else in that slot is public data an earlier
shape left there, to be re-rendered rather than trusted as a snapshot.

It is a store record like any other, so it carries the store's one schema and
one version — see ``lib/store.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT, accept
from ltspice_mcp.lib.store import envelope as _store_envelope

logger = logging.getLogger(__name__)

_SOURCE = "<attached-analysis-snapshot>"


def envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one neutral snapshot so it cannot be mistaken for a public result."""
    return _store_envelope(KIND_ANALYSIS_SNAPSHOT, **payload)


def classify(value: Any) -> Literal["snapshot", "unsupported", "legacy"]:
    """Say what a stored attached-analysis value is.

    ``legacy`` means the slot holds something that was never a snapshot — a
    public analysis result written before the envelope existed. ``unsupported``
    means it IS a snapshot, from a store version this build does not read.
    Either way the renderer refuses and points at re-running the analysis:
    guessing at a shape this build did not write is what the envelope exists to
    prevent.
    """
    if not isinstance(value, dict) or value.get("kind") != KIND_ANALYSIS_SNAPSHOT:
        return "legacy"
    return (
        "snapshot"
        if accept(value, _SOURCE, kind=KIND_ANALYSIS_SNAPSHOT, log=logger)
        else "unsupported"
    )
