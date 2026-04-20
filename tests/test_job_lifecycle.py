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
from pathlib import Path

import pytest

from ltspice_mcp.lib import now
from ltspice_mcp.lib.job_lifecycle import (
    STATUS_TO_EVENT,
    TERMINAL_STATUSES,
    VALID_BATCH_TRANSITIONS,
    VALID_SIM_TRANSITIONS,
    InvalidTransitionError,
    recover,
    transition,
)
from ltspice_mcp.state import BatchJob, SimulationJob


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


def _sim_job(status: str = "queued", tmp_path: Path | None = None) -> SimulationJob:
    circuit = (tmp_path or Path("/tmp")) / "rc.cir"
    return SimulationJob(
        job_id="sim_lifecycle",
        netlist=circuit,
        simulator="LTspice",
        status=status,  # type: ignore[arg-type]
        started_at=now(),
    )


def _batch_job(status: str = "running", tmp_path: Path | None = None) -> BatchJob:
    circuit = (tmp_path or Path("/tmp")) / "amp.cir"
    return BatchJob(
        job_id="batch_lifecycle",
        job_type="sweep",
        netlist=circuit,
        total_runs=5,
        status=status,  # type: ignore[arg-type]
    )


class TestTransitionSim:
    def test_queued_to_running_emits_started(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        job = _sim_job("queued", tmp_path)
        transition(job, "running", simulator="LTspice")
        assert job.status == "running"
        assert job.completed_at is None  # not terminal
        events = _events(events_caplog)
        assert len(events) == 1
        assert events[0]["event"] == "started"
        assert events[0]["simulator"] == "LTspice"

    def test_running_to_completed_sets_completed_at_and_done_event(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        job = _sim_job("running", tmp_path)
        assert not job.done_event.is_set()
        transition(job, "completed", raw_size_bytes=1024)
        assert job.status == "completed"
        assert job.completed_at is not None
        assert job.done_event.is_set()
        e = _events(events_caplog)[-1]
        assert e["event"] == "completed"
        assert e["raw_size_bytes"] == 1024

    def test_running_to_failed_emits_failed(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        job = _sim_job("running", tmp_path)
        transition(job, "failed", error="boom", phase="execution")
        assert job.status == "failed"
        e = _events(events_caplog)[-1]
        assert e["event"] == "failed"
        assert e["error"] == "boom"
        assert e["phase"] == "execution"

    def test_running_to_timeout_emits_failed_event(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """timeout is a status but the emitted event is 'failed'."""
        job = _sim_job("running", tmp_path)
        transition(job, "timeout", duration_s=30)
        assert job.status == "timeout"
        e = _events(events_caplog)[-1]
        assert e["event"] == "failed"

    def test_invalid_transition_raises(self, tmp_path: Path) -> None:
        job = _sim_job("completed", tmp_path)
        with pytest.raises(InvalidTransitionError, match="illegal transition"):
            transition(job, "running")

    def test_same_status_raises(self, tmp_path: Path) -> None:
        job = _sim_job("running", tmp_path)
        with pytest.raises(InvalidTransitionError, match="no-op"):
            transition(job, "running")

    def test_terminal_state_has_no_outgoing(self, tmp_path: Path) -> None:
        for terminal in ("completed", "failed", "cancelled", "timeout"):
            job = _sim_job(terminal, tmp_path)
            with pytest.raises(InvalidTransitionError):
                transition(job, "running")


class TestTransitionBatch:
    def test_running_to_completed(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        bj = _batch_job("running", tmp_path)
        transition(bj, "completed", completed_runs=5, total_runs=5)
        assert bj.status == "completed"
        assert bj.done_event.is_set()
        e = _events(events_caplog)[-1]
        assert e["event"] == "completed"

    def test_running_to_cancelled(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        bj = _batch_job("running", tmp_path)
        transition(bj, "cancelled", completed_runs=2, total_runs=5)
        assert bj.status == "cancelled"
        e = _events(events_caplog)[-1]
        assert e["event"] == "cancelled"

    def test_batch_has_no_queued_state(self, tmp_path: Path) -> None:
        bj = _batch_job("running", tmp_path)
        with pytest.raises(InvalidTransitionError):
            transition(bj, "queued")

    def test_interrupted_batch_terminal(self, tmp_path: Path) -> None:
        bj = _batch_job("interrupted", tmp_path)
        # Batch interrupted has NO outgoing edges (unlike sim).
        with pytest.raises(InvalidTransitionError):
            transition(bj, "completed")


class TestRecover:
    def test_interrupted_to_completed_emits_recovered_event(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        job = _sim_job("interrupted", tmp_path)
        recover(job, "completed")
        assert job.status == "completed"
        e = _events(events_caplog)[-1]
        assert e["event"] == "interrupted_recovered"
        assert e["recovered_as"] == "completed"

    def test_non_interrupted_rejected(self, tmp_path: Path) -> None:
        job = _sim_job("running", tmp_path)
        with pytest.raises(InvalidTransitionError, match="requires current status"):
            recover(job, "completed")


class TestStateMachineStructure:
    """Static consistency checks — catch table rot at test time, not in prod."""

    def test_every_transition_target_has_event_mapping(self) -> None:
        """Every status reachable as a transition target must map to an event.

        'interrupted' is special — it's reached only via persistence
        deserialization (job_store._finalize_loaded_status), not via
        ``transition()``. The recovery path out of interrupted uses
        ``recover()`` which has its own event name. So 'interrupted'
        doesn't need a STATUS_TO_EVENT entry.
        """
        special = {"interrupted"}
        for source, targets in VALID_SIM_TRANSITIONS.items():
            for target in targets:
                if target in special:
                    continue
                assert target in STATUS_TO_EVENT, (
                    f"sim transition {source} → {target} lands on a status "
                    f"with no event mapping; add '{target}' to STATUS_TO_EVENT"
                )
        for source, targets in VALID_BATCH_TRANSITIONS.items():
            for target in targets:
                if target in special:
                    continue
                assert target in STATUS_TO_EVENT, (
                    f"batch transition {source} → {target} lands on a status "
                    f"with no event mapping"
                )

    def test_terminal_statuses_have_no_outgoing(self) -> None:
        """TERMINAL_STATUSES must not appear as sources with outgoing edges.

        Exception: 'interrupted' is terminal but the sim table allows
        recovery out of it via the ``recover()`` path.
        """
        for source, targets in VALID_SIM_TRANSITIONS.items():
            if source in TERMINAL_STATUSES and source != "interrupted":
                assert not targets, (
                    f"sim terminal status {source} has outgoing edges: {targets}"
                )
        for source, targets in VALID_BATCH_TRANSITIONS.items():
            if source in TERMINAL_STATUSES:
                assert not targets, (
                    f"batch terminal status {source} has outgoing edges: {targets}"
                )

    def test_no_status_writes_outside_lifecycle_module(self) -> None:
        """Production code must not mutate ``job.status`` directly.

        The chokepoint is ``transition()`` / ``recover()``. This test
        greps the src tree (excluding job_lifecycle.py itself, which
        defines the chokepoint) and fails if any file still writes
        status directly.
        """
        import ast

        root = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"
        offenders: list[str] = []
        for py in root.rglob("*.py"):
            if py.name == "job_lifecycle.py":
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # Match: `something.status = X` (Assign or AnnAssign)
                if isinstance(node, ast.Assign):
                    targets: list[ast.expr] = list(node.targets)
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                else:
                    continue
                for t in targets:
                    if isinstance(t, ast.Attribute) and t.attr == "status":
                        offenders.append(
                            f"  {py.relative_to(root.parent.parent)}:"
                            f"{node.lineno}: {ast.unparse(node)}"
                        )
        assert not offenders, (
            f"{len(offenders)} direct `.status = …` write(s) found outside "
            f"job_lifecycle.py:\n" + "\n".join(offenders) + "\n"
            "Route status changes through transition() or recover() instead."
        )
