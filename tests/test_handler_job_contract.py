"""Cross-status CONTRACT test for the jobs reader.

Closes the bug class where a fix lands for one variant (one status) while
sibling variants stay broken. Asserted through the REAL handler, not an
internal, so a formatter that grew a hardcoded status allowlist fails here.

The original instance: the status formatter carried an allowlist that omitted
``interrupted`` — a status only assigned on restart recovery, so no happy-path
test reached it, and asking about a recovered job raised "unexpected status"
instead of reporting one. Parametrizing over every terminal status catches the
class rather than the one instance.
"""

from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.state import TERMINAL_STATUSES, SessionState
from ltspice_mcp.tools.experiments import JobsInput, handle_jobs

pytestmark = pytest.mark.asyncio

# Every terminal status an experiment can hold. Derived from the shared set so a
# new terminal status propagates into this contract instead of escaping it.
EXPERIMENT_TERMINAL_STATUSES = sorted(TERMINAL_STATUSES)


def _make_experiment(state: SessionState, *, status: str) -> ExperimentJob:
    circuit = Path(state.working_dir) / "deck.cir"
    circuit.write_text(".op\n.end\n", encoding="utf-8")
    job = ExperimentJob(
        job_id="exp_status",
        request_id="request-status",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=experiment_store.record_path("exp_status", Path(state.working_dir)),
        cases=[
            ExperimentCase(
                case_id="case_0000",
                run_index=0,
                circuit="dut",
                circuit_path=circuit,
                staged_deck=circuit,
                deck_sha256="a" * 64,
                assignments={},
                status="produced",
            )
        ],
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
        completeness=Completeness(declared=1, expanded=1, produced=1),
        status=status,  # type: ignore[arg-type]
    )
    state.add_experiment_job(job, already_persisted=True)
    return job


class TestTerminalStatusCompleteness:
    @pytest.mark.parametrize("status", EXPERIMENT_TERMINAL_STATUSES)
    async def test_jobs_status_handles_every_terminal_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_experiment(state_no_sim, status=status)
        result = await handle_jobs(
            JobsInput.model_validate({"action": "status", "job_id": "exp_status"}),
            state_no_sim,
        )
        data = result.structuredContent
        assert data is not None
        text = result.content[0].text  # type: ignore[union-attr]
        assert "unexpected status" not in text.lower()
        # The status is surfaced to the caller, not swallowed.
        assert data["status"] == status
