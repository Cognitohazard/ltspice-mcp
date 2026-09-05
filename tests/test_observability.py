"""Tests for structured job lifecycle event emission.

The process's stderr logger is the only channel these events have: the MCP
logging capability that could once relay them to the client is deprecated as
of the 2026-07-28 revision and is not served.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState


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
    return SessionState.create(config, available={})


def _events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """Structured event payloads from the captured records."""
    return [
        record.ltspice_event  # type: ignore[attr-defined]
        for record in caplog.records
        if hasattr(record, "ltspice_event")
    ]


def _experiment(working_dir: Path, circuit: Path, *, job_id: str, status: str) -> ExperimentJob:
    case = ExperimentCase(
        case_id="case_0000",
        run_index=0,
        circuit="dut",
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="a" * 64,
        assignments={},
        status="queued",
    )
    return ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=Store(working_dir).job_record(job_id),
        cases=[case],
        sources=[
            SourceRecord(
                circuit="dut",
                path=circuit,
                sha256="b" * 64,
                staged_deck=circuit,
                manifest=[],
                simulator="FakeSim",
                dialect="ltspice",
            )
        ],
        simulator="FakeSim",
        completeness=Completeness(declared=1, expanded=1),
        status=status,  # type: ignore[arg-type]
    )


class TestEmitJobEvent:
    def test_emits_structured_payload(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_test_001", status="queued")
        emit_job_event("submitted", job, simulator="FakeSim")

        events = _events(events_caplog)
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "submitted"
        assert event["kind"] == "experiment"
        assert event["job_id"] == "exp_test_001"
        assert event["sources"] == [str(circuit)]
        assert event["simulator"] == "FakeSim"
        assert "ts" in event
        assert "duration_s" in event

    def test_duration_on_completion(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_dur", status="completed")
        emit_job_event("completed", job)
        event = _events(events_caplog)[-1]
        # duration >= 0 (started_at defaults to now)
        assert event["duration_s"] is not None
        assert event["duration_s"] >= 0

    def test_extra_kwargs_merge_into_payload(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_extra", status="running")
        emit_job_event("started", job, expanded=3, note="hello")
        event = _events(events_caplog)[-1]
        assert event["expanded"] == 3
        assert event["note"] == "hello"

    def test_payload_is_json_serialisable(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        # The payload rides a log record into whatever handler the host
        # installs; a Path left unstringified there fails at the sink, far from
        # the emit that produced it.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_json", status="running")
        emit_job_event("started", job)
        json.dumps(_events(events_caplog)[-1])


class TestJobRegistryEmitsOnAdd:
    def test_registering_an_experiment_emits_submitted(
        self, state: SessionState, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_add", status="queued")
        state.add_experiment_job(job, already_persisted=True)

        event = _events(events_caplog)[-1]
        assert event["event"] == "submitted"
        assert event["kind"] == "experiment"
        assert event["job_id"] == "exp_add"
