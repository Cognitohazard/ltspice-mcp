"""The status vocabulary every job record is written and read against.

This module is the leaf of the job-subsystem dependency graph: it defines the
vocabulary and nothing else. Extracting it from ``state.py`` broke a cluster of
import cycles where ``state`` imported its collaborators (``lib.job_registry``,
``lib.job_lifecycle``, ``lib.observability``) and those collaborators needed
the definitions back for their type signatures.

After this split, the graph is strictly layered:

    lib.job_types (this file)
        ↑
        ├── state (SessionState composes JobRegistry)
        ├── lib.job_registry
        ├── lib.job_lifecycle (transition chokepoint)
        └── lib.observability (event emission)

``state.py`` re-exports both names, so a caller can read the vocabulary from
either place.

Both sets are derived from the two halves of ``ExperimentStatus``, and no other
module spells a set of job statuses: a status added to the record's vocabulary
is live or terminal everywhere at once, and one the vocabulary lacks is neither,
so a record carrying it is refused when it is read.
"""

from __future__ import annotations

from typing import get_args

from ltspice_mcp.lib.experiment_types import LiveExperimentStatus, TerminalExperimentStatus

# Terminal statuses — eligible for eviction, represent finished work.
# 'interrupted' is terminal because the owning runner is gone; metadata
# and any partial outputs are preserved but the job cannot resume
# in-process (recovery promotes it via lib.job_lifecycle.recover).
TERMINAL_STATUSES: frozenset[str] = frozenset(get_args(TerminalExperimentStatus))

# Statuses that only make sense while a runner owns the job. Seeing one
# in a persisted record means the prior server died mid-run.
NON_TERMINAL_LIVE_STATUSES: frozenset[str] = frozenset(get_args(LiveExperimentStatus))
