"""The envelope around a durable attached-analysis result.

An experiment's attached analysis is computed once and stored inside the job
record, where it sits next to values a caller reads directly. The envelope is
what keeps the two apart: a stored analysis carrying it is this build's own
neutral snapshot, and anything else in that slot is public data an earlier
shape left there, to be re-rendered rather than trusted as a snapshot.

It is a store record like any other, so it carries the store's one schema and
one version — see ``lib/store.py`` — and, inside it, the version of the
analysis it holds (:data:`ANSWER_VERSION`). :func:`classify` is the one
answer to whether this build renders a stored analysis; the store version
says whether this build reads the record at all, this one what the record's
analysis is, since a release can change the analysis it stores without
changing the store.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Literal

from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT, accept
from ltspice_mcp.lib.store import envelope as _store_envelope

logger = logging.getLogger(__name__)

_SOURCE = "<attached-analysis-snapshot>"


#: The version of the analysis this build stores and renders: its complete
#: answer, the snapshot it was computed from and its request. The release
#: before stored a snapshot of its own assembly under the same store version,
#: with no such field.
ANSWER_VERSION: Final = 2


def envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one neutral snapshot, at :data:`ANSWER_VERSION`, so it cannot be
    mistaken for a public result."""
    return _store_envelope(KIND_ANALYSIS_SNAPSHOT, answer_version=ANSWER_VERSION, **payload)


def classify(value: Any) -> Literal["snapshot", "unsupported", "legacy"]:
    """Say what a stored attached-analysis value is.

    ``snapshot`` means this build renders it: a record of the store version it
    reads, holding an analysis at :data:`ANSWER_VERSION`. ``legacy`` means the
    slot holds what an earlier release wrote there: a public analysis result
    written before the envelope existed, or a snapshot of an earlier
    release's own assembly (no :data:`ANSWER_VERSION`). ``unsupported`` means
    a record of a store version this build does not read. Every way but the
    first the renderer refuses and points at re-running the analysis:
    guessing at a shape this build did not write is what the envelope exists
    to prevent.
    """
    if not isinstance(value, dict) or value.get("kind") != KIND_ANALYSIS_SNAPSHOT:
        return "legacy"
    if not accept(value, _SOURCE, kind=KIND_ANALYSIS_SNAPSHOT, log=logger):
        return "unsupported"
    return "snapshot" if value.get("answer_version") == ANSWER_VERSION else "legacy"
