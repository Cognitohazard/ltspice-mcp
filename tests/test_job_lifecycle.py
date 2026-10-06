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

import ast
import logging
from pathlib import Path

import pytest

from ltspice_mcp.lib.experiment_types import Completeness, ExperimentJob
from ltspice_mcp.lib.job_lifecycle import (
    STATUS_TO_EVENT,
    TERMINAL_STATUSES,
    VALID_EXPERIMENT_TRANSITIONS,
    InvalidTransitionError,
    transition,
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

        'interrupted' is special — it is reached only when a record is loaded
        back from disk and its owner is gone (experiment_store._reconcile_restart),
        never through ``transition()``, so it needs no STATUS_TO_EVENT entry.
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


def _job(status: str) -> ExperimentJob:
    return ExperimentJob(
        job_id="exp_lifecycle_0001",
        request_id="lifecycle-request",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=Path("exp_lifecycle_0001.json"),
        cases=[],
        sources=[],
        simulator="FakeSim",
        completeness=Completeness(declared=0, expanded=0),
        status=status,  # type: ignore[arg-type]
    )


_RUNTIME_EDGES = [
    (source, target)
    for source, targets in VALID_EXPERIMENT_TRANSITIONS.items()
    for target in sorted(targets)
    if target in STATUS_TO_EVENT
]


class TestTransitionEvents:
    """Every legal transition emits one event; a refused one emits none."""

    @pytest.mark.parametrize(("source", "target"), _RUNTIME_EDGES)
    def test_each_legal_transition_emits_one_named_event(
        self, events_caplog: pytest.LogCaptureFixture, source: str, target: str
    ) -> None:
        job = _job(source)
        transition(job, target, reason="lifecycle test")

        assert job.status == target
        (event,) = _events(events_caplog)
        assert event["event"] == STATUS_TO_EVENT[target]
        assert event["job_id"] == job.job_id
        assert event["reason"] == "lifecycle test"

    @pytest.mark.parametrize(
        ("source", "target"),
        [("completed", "running"), ("running", "running"), ("running", "interrupted")],
        ids=["out-of-table", "same-status", "restart-only"],
    )
    def test_a_refused_transition_emits_nothing(
        self, events_caplog: pytest.LogCaptureFixture, source: str, target: str
    ) -> None:
        job = _job(source)
        with pytest.raises(InvalidTransitionError):
            transition(job, target)

        assert job.status == source
        assert _events(events_caplog) == []


_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"

# Where a status may change without going through a live job: the live job's
# own transition, and shutdown settling a job this process holds no live job for.
_BARE_TRANSITIONS = {
    ("lib/job_lifecycle.py", "LiveJob.transition"),
    ("lib/job_registry.py", "JobRegistry.cancel_running"),
}


class _ScopedCalls(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.calls: list[tuple[str, ast.Call]] = []

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((".".join(self.scope), node))
        self.generic_visit(node)


def _is_bare_transition(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "transition"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "transition"
        and isinstance(func.value, ast.Name)
        and func.value.id == "job_lifecycle"
    )


def test_a_running_jobs_status_changes_through_its_live_job() -> None:
    """A bare ``transition()`` on a job this process runs would change the
    record and leave everyone waiting on its live job waiting, so the status
    of a job with a live job changes only through ``LiveJob.transition``."""
    found = set()
    for path in sorted(_SRC.rglob("*.py")):
        visitor = _ScopedCalls()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= {
            (path.relative_to(_SRC).as_posix(), scope)
            for scope, call in visitor.calls
            if _is_bare_transition(call)
        }
    assert found == _BARE_TRANSITIONS
