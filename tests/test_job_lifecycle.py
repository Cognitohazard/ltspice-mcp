"""Tests for the job lifecycle state machine.

Two layers of guarantee:

1. Behaviorally: ``transition()`` advances job state only via edges
   listed in the transition tables, and every legal transition emits
   exactly one lifecycle event with the name from ``STATUS_TO_EVENT``.
2. Structurally: the transition tables and the event mapping agree —
   every status reachable as a transition target has an event mapping,
   and event-emitting statuses are reachable.
"""

from __future__ import annotations

import logging

import pytest

from ltspice_mcp.lib.job_lifecycle import (
    STATUS_TO_EVENT,
    TERMINAL_STATUSES,
    VALID_EXPERIMENT_TRANSITIONS,
)


def _events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        r.__dict__["ltspice_event"]
        for r in caplog.records
        if r.name == "ltspice_mcp.events" and hasattr(r, "ltspice_event")
    ]


@pytest.fixture
def events_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="ltspice_mcp.events")
    return caplog


class TestStateMachineStructure:
    """Static consistency checks — catch table rot at test time, not in prod."""

    def test_every_transition_target_has_event_mapping(self) -> None:
        """Every status reachable as a transition target must map to an event.

        'interrupted' is special — it is reached only through persistence
        deserialization (job_store._finalize_loaded_status), never through
        ``transition()``, so it needs no STATUS_TO_EVENT entry.
        """
        special = {"interrupted"}
        for source, targets in VALID_EXPERIMENT_TRANSITIONS.items():
            for target in targets:
                if target in special:
                    continue
                assert target in STATUS_TO_EVENT, (
                    f"experiment transition {source} → {target} lands on a "
                    "status with no event mapping"
                )

    def test_terminal_statuses_have_no_outgoing(self) -> None:
        """TERMINAL_STATUSES must not appear as sources with outgoing edges."""
        for source, targets in VALID_EXPERIMENT_TRANSITIONS.items():
            if source in TERMINAL_STATUSES:
                assert not targets, (
                    f"experiment terminal status {source} has outgoing edges: {targets}"
                )

    def test_every_event_status_is_reachable(self) -> None:
        """Every status with an event mapping must be a reachable target —
        an unreachable entry is dead vocabulary that reads as supported."""
        reachable = {
            target for targets in VALID_EXPERIMENT_TRANSITIONS.values() for target in targets
        }
        for status in STATUS_TO_EVENT:
            assert status in reachable, (
                f"STATUS_TO_EVENT names {status!r}, which no transition reaches"
            )
