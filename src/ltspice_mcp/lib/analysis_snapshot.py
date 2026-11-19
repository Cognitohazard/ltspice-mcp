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

from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT, STORE_VERSION, accept
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


Verdict = Literal["snapshot", "earlier", "later", "damaged"]


def classify(value: Any) -> Verdict:
    """Say what a stored attached-analysis value is: the one answer to
    whether this build renders it.

    ``snapshot``: this build renders it, a record of a store version it reads
    holding an analysis at :data:`ANSWER_VERSION` whose answer is an object.
    ``earlier``: an earlier release wrote it, a public analysis result
    written before the envelope existed, a record of an older store version,
    or an analysis of an older version (none, before versions were written).
    ``later``: a later release wrote it, a newer store or analysis version.
    ``damaged``: a record of this version whose analysis cannot be rendered.
    Every way but the first the renderer refuses and says which: guessing at
    a shape this build did not write is what the envelope exists to prevent.
    """
    if not isinstance(value, dict) or value.get("kind") != KIND_ANALYSIS_SNAPSHOT:
        return "earlier"
    store_version = value.get("store_version")
    if not isinstance(store_version, int) or isinstance(store_version, bool):
        return "damaged"
    if store_version > STORE_VERSION:
        return "later"
    if not accept(value, _SOURCE, kind=KIND_ANALYSIS_SNAPSHOT, log=logger):
        return "earlier" if store_version < STORE_VERSION else "damaged"
    answer_version = value.get("answer_version", 0)
    if not isinstance(answer_version, int) or isinstance(answer_version, bool):
        return "damaged"
    if answer_version < ANSWER_VERSION:
        return "earlier"
    if answer_version > ANSWER_VERSION:
        return "later"
    return "snapshot" if isinstance(value.get("answer"), dict) else "damaged"
