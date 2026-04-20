"""Tests for structured job lifecycle event emission."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob


@pytest.fixture
def events_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture records from the dedicated events logger."""
    caplog.set_level(logging.INFO, logger="ltspice_mcp.events")
    return caplog


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    config = ServerConfig(
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        persist_jobs=False,
        log_level="DEBUG",
    )
    return SessionState.create(config, {})


def _events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """Extract the structured payloads attached to event log records."""
    return [
        r.__dict__["ltspice_event"]
        for r in caplog.records
        if r.name == "ltspice_mcp.events" and hasattr(r, "ltspice_event")
    ]


class TestEmitJobEvent:
    def test_emits_structured_payload(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_test_001",
            netlist=circuit,
            simulator="LTspice",
            status="queued",
            started_at=now(),
        )
        emit_job_event("submitted", job, simulator="LTspice")

        events = _events(events_caplog)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "submitted"
        assert e["kind"] == "sim"
        assert e["job_id"] == "sim_test_001"
        assert e["netlist"] == str(circuit)
        assert e["simulator"] == "LTspice"
        assert "ts" in e
        assert "duration_s" in e

    def test_duration_on_completion(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_dur",
            netlist=circuit,
            simulator="LTspice",
            status="completed",
            started_at=now(),
        )
        emit_job_event("completed", job)
        e = _events(events_caplog)[-1]
        # duration >= 0 (we just set started_at=now())
        assert e["duration_s"] is not None
        assert e["duration_s"] >= 0

    def test_batch_job_kind_inferred(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        bj = BatchJob(
            job_id="batch_sweep_1",
            job_type="sweep",
            netlist=circuit,
            total_runs=5,
        )
        emit_job_event("submitted", bj, total_runs=5)
        e = _events(events_caplog)[-1]
        assert e["kind"] == "sweep"

        bj_mc = BatchJob(
            job_id="batch_mc_1",
            job_type="montecarlo",
            netlist=circuit,
            total_runs=100,
        )
        emit_job_event("submitted", bj_mc, total_runs=100)
        e = _events(events_caplog)[-1]
        assert e["kind"] == "montecarlo"

    def test_extra_kwargs_merged(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_err",
            netlist=circuit,
            simulator="LTspice",
            status="failed",
            started_at=now(),
        )
        emit_job_event("failed", job, error="No simulator", phase="submission")
        e = _events(events_caplog)[-1]
        assert e["error"] == "No simulator"
        assert e["phase"] == "submission"


class TestJobRegistryEmitsOnAdd:
    def test_sim_job_submission_emits_event(
        self, events_caplog: pytest.LogCaptureFixture, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_add",
            netlist=circuit,
            simulator="LTspice",
            status="queued",
            started_at=now(),
        )
        state.add_job(job)
        events = [e for e in _events(events_caplog) if e["job_id"] == "sim_add"]
        assert any(e["event"] == "submitted" and e["kind"] == "sim" for e in events)

    def test_batch_job_submission_emits_event(
        self, events_caplog: pytest.LogCaptureFixture, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        bj = BatchJob(
            job_id="batch_add",
            job_type="sweep",
            netlist=circuit,
            total_runs=3,
        )
        state.add_batch_job(bj)
        events = [e for e in _events(events_caplog) if e["job_id"] == "batch_add"]
        assert any(e["event"] == "submitted" and e["kind"] == "sweep" for e in events)
