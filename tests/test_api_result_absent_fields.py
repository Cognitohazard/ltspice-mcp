"""A Python API result answers a field it left out with that field's empty value.

A reply leaves out every field that says nothing (``lib/reply_fields.py``), and
the Python API's results keep that shape. Reading such a field answers the
empty value of its declared type, so a script need not guard every read; a key
the operation never declares still raises, and the result serialises exactly
as the sent dict does.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import ApiCallError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import LIVENESS_S, SyncApi, fake_simulator

pytestmark = pytest.mark.asyncio


def _rc_deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text(
        "* rc\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n", encoding="utf-8"
    )
    return deck


async def _verify(state: SessionState, work_dir: Path) -> dict[str, Any]:
    args = {"path": str(_rc_deck(work_dir)), "checks": ["syntax", "quality"]}
    return await asyncio.to_thread(lambda: SyncApi(state).verify_circuit(**args))


async def test_a_left_out_list_reads_as_empty(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    assert "failures" not in result
    assert result["failures"] == []
    assert result.get("failures") == []
    # A caller's own default is still the caller's.
    assert result.get("failures", "none") == "none"
    assert "failures" not in result, "reading a left-out field must not add it"


async def test_a_misspelt_key_still_raises(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    with pytest.raises(KeyError):
        result["failurez"]
    assert result.get("failurez") is None


async def test_the_result_serialises_as_the_sent_dict(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    plain = json.loads(json.dumps(result))
    assert json.dumps(result) == json.dumps(plain)
    assert json.dumps(result, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert repr(result) == repr(plain)
    assert result == plain
    assert list(result) == list(plain)
    assert len(result) == len(plain)
    assert type(dict(result)) is dict
    assert type(result.copy()) is dict


def _run_args(work_dir: Path, request_id: str) -> dict[str, Any]:
    deck = work_dir / "op.cir"
    deck.write_text("* deck\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n", encoding="utf-8")
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
    }


async def test_a_run_row_answers_its_left_out_assignments(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A run with no sweep has no assignments; every route that hands a run
    row to a script answers the left-out field the same way."""
    fake_simulator(monkeypatch)
    arguments = _run_args(work_dir, "absent-assignments")
    # The run itself goes on this test's loop, which outlives it; each API
    # call below runs on a loop of its own.
    ran = await handle_run_experiments(
        RunExperimentsInput.model_validate(arguments), state_with_sim
    )
    assert ran.structured_content is not None
    assert ran.structured_content["status"] == "completed", ran.structured_content
    api = SyncApi(state_with_sim)
    # The same request again replays the finished job's receipt.
    receipt = await asyncio.to_thread(lambda: api.run_experiments(**arguments))
    job_id = receipt["job_id"]
    status = await asyncio.to_thread(lambda: api.jobs(action="status", job_id=job_id))
    waited = await asyncio.to_thread(lambda: api.wait(job_id, timeout_s=LIVENESS_S))
    raw = await asyncio.to_thread(lambda: api.jobs(action="status", job_id=job_id, raw_page=True))
    for result in (receipt, status, waited, raw):
        rows = result["runs"]["items"]
        assert rows
        for row in rows:
            assert "assignments" not in row
            assert row["assignments"] == {}
            assert row.get("assignments") == {}
            with pytest.raises(KeyError):
                row["assignmentz"]


async def test_an_error_payload_answers_its_left_out_fields(state_no_sim: SessionState):
    with pytest.raises(ApiCallError) as caught:
        await asyncio.to_thread(
            lambda: SyncApi(state_no_sim).jobs(action="status", job_id="no-such-job")
        )
    payload = caught.value.payload
    assert "observations" not in payload
    assert payload["observations"] == []
    assert payload.get("observations") == []
    with pytest.raises(KeyError):
        payload["observationz"]
