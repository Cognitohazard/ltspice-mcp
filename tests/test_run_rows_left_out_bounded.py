"""``runs_not_shown`` has one bound in tokens, and the size search keeps as
many run rows as fit beside it.

Design principle "A short reply, the data behind an id": a reply that leaves
rows out names the facts on them in a part that is never cut, so that part is
bounded by itself, whatever the number of kinds and runs. The rows are then
sized against what that part leaves of the target.
"""

from __future__ import annotations

import random
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import log_conditions, now
from ltspice_mcp.lib.presenter import Record, present
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import jobs as jobs_module
from ltspice_mcp.tools import receipts
from ltspice_mcp.tools.jobs import (
    JOBS_OUTPUT_SCHEMA,
    JobsEvaluation,
    JobsInput,
    JobsResumeInput,
    handle_jobs,
)
from ltspice_mcp.tools.receipts import RUN_EXPERIMENTS_OUTPUT_SCHEMA, snapshot_receipt
from tests.conftest import make_experiment_job

#: Every diagnostic code a log rule reports.
_CODES = sorted({rule.code for rule in log_conditions._RULES if rule.code})


async def _jobs(state: SessionState, **values: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(values), state)
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


def _diagnosed(codes: list[str]) -> dict[str, Any]:
    return {
        "diagnostics": [
            {"code": code, "count": 1, "first_text": code, "channel": "log"} for code in codes
        ]
    }


def _job(state: SessionState, job_id: str, count: int) -> Any:
    job = make_experiment_job(state, job_id=job_id, count=count)
    for case in job.cases:
        case.submitted_at = now()
        case.log_diagnostics = {}
        case.measurement_facts = {}
    return job


# ---------------------------------------------------------------------------
# 1. One bound
# ---------------------------------------------------------------------------


def test_the_bound_is_a_few_hundred_tokens():
    assert 0 < receipts.NOT_SHOWN_BOUND_TOKENS <= 700


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 1500])
async def test_runs_not_shown_keeps_its_bound_and_every_kinds_count(
    state_no_sim: SessionState, budget: int | None
):
    job = _job(state_no_sim, "exp_kinds", 2000)
    for index, case in enumerate(job.cases):
        case.log_diagnostics = _diagnosed([_CODES[index % len(_CODES)]])
    extra = {} if budget is None else {"budget": budget}
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, **extra)

    shown = len(data["runs"]["items"])
    note = data["runs_not_shown"]
    assert estimate_tokens({"runs_not_shown": note}) <= receipts.NOT_SHOWN_BOUND_TOKENS
    expected: dict[str, int] = {}
    for case in job.cases[shown:]:
        for entry in case.log_diagnostics["diagnostics"]:
            kind = f"diagnostics.{entry['code']}"
            expected[kind] = expected.get(kind, 0) + 1
    assert note["counts"] == expected
    assert 1 <= len(note["runs"]) <= receipts.NOT_SHOWN_NAMED


@pytest.mark.asyncio
async def test_a_kind_every_run_carries_is_counted_and_names_no_run(state_no_sim: SessionState):
    job = _job(state_no_sim, "exp_every", 300)
    for case in job.cases:
        case.measurement_facts = {"coverage": "simulator_unsupported"}
    job.cases[250].log_diagnostics = _diagnosed([_CODES[0]])
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=600)

    note = data["runs_not_shown"]
    left = 300 - len(data["runs"]["items"])
    assert note["counts"]["measurements_coverage.simulator_unsupported"] == left
    named = [kind for run in note["runs"] for kind in run["kinds"]]
    assert "measurements_coverage.simulator_unsupported" not in named
    assert named == [f"diagnostics.{_CODES[0]}"]


@pytest.mark.asyncio
async def test_the_over_target_note_names_runs_not_shown(state_no_sim: SessionState):
    job = _job(state_no_sim, "exp_over", 400)
    for index, case in enumerate(job.cases):
        case.log_diagnostics = _diagnosed(_CODES[: 1 + index % len(_CODES)])
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=500)
    (over,) = [item for item in data.get("observations", []) if item["code"] == "budget_not_met"]
    assert "runs_not_shown" in over["detail"]


# ---------------------------------------------------------------------------
# 2. The size search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [3500, 4000, 5000])
async def test_rows_that_fit_beside_a_large_note_are_kept(state_no_sim: SessionState, budget: int):
    """Rows 33 and 34 carry every diagnostic code: a reply that keeps fewer
    names them all, and the search must still find the many rows that fit."""
    job = _job(state_no_sim, f"exp_search_{budget}", 200)
    for index in (33, 34):
        job.cases[index].log_diagnostics = _diagnosed(_CODES)
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=budget)

    assert not [item for item in data.get("observations", []) if item["code"] == "budget_not_met"]
    assert estimate_tokens(data) <= budget
    assert len(data["runs"]["items"]) > 20


_SCHEMA = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {"type": "string"}, "x-cut": "items"}},
}


def _note(facts: list[int], start: int) -> list[str]:
    """A note on the facts left out, which shrinks as more rows are kept, and
    is bounded: at most 12 entries of 40 characters."""
    return [f"fact at {index:04d} " + "z" * 26 for index in facts if index >= start][:12]


@pytest.mark.parametrize("seed", range(40))
def test_the_search_keeps_every_row_that_fits_beside_the_reserved_note(tmp_path, seed: int):
    rng = random.Random(seed)
    count = rng.randint(2, 120)
    rows = [f"row {index:04d} " + "y" * rng.randint(4, 60) for index in range(count)]
    facts = sorted(rng.sample(range(count), rng.randint(0, min(count, 20))))
    reserve = estimate_tokens({"left_out": ["fact at 0000 " + "z" * 26] * 12})

    def annotate(reply: dict[str, Any], cut: Any, _stored: Any, _paged: Any) -> None:
        entry = cut.get("/rows")
        note = _note(facts, 0 if entry is None else entry.returned)
        if note:
            reply["left_out"] = note

    def presented(target: int, cap: int | None = None) -> Any:
        return present(
            {"rows": rows},
            _SCHEMA,
            target=target,
            tool="tool",
            snapshot={},
            store=Store(tmp_path),
            ttl_hours=1.0,
            annotate=annotate,
            records={"/rows": Record("rows", "r")},
            caps=None if cap is None else {"/rows": cap},
            apart={"left_out": reserve},
        )

    whole = estimate_tokens({"rows": rows}, as_sent=False)
    target = rng.randint(40, max(41, whole))
    shown = presented(target)
    kept = len(shown.reply["rows"])

    # Each count's reply, whole and without the note, as the search sizes it.
    full = {c: presented(10**9, cap=c).reply for c in range(1, count + 1)}
    size = {c: estimate_tokens(full[c], as_sent=False) for c in full}
    bare = {
        c: estimate_tokens({k: v for k, v in full[c].items() if k != "left_out"}, as_sent=False)
        for c in full
    }
    if not shown.met:
        assert not [c for c in full if size[c] <= target], "a count that fits was missed"
    else:
        assert kept == count or bare[kept + 1] + reserve > target or size[kept + 1] > target


# ---------------------------------------------------------------------------
# 3. The lint branch
# ---------------------------------------------------------------------------


def test_a_whole_lint_entry_has_findings():
    lint = RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["lint"]["items"]
    validator = jsonschema.Draft202012Validator(lint)
    assert not validator.is_valid({"circuit": "dut"})
    assert validator.is_valid({"circuit": "dut", "said_before": 1})


# ---------------------------------------------------------------------------
# 4. A replayed resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_replayed_resume_sends_every_note_in_full(state_no_sim: SessionState):
    job = _job(state_no_sim, "exp_resume_replay", 2)
    await _jobs(state_no_sim, action="wait", job_id=job.job_id, timeout_s=0)
    args = JobsResumeInput.model_validate(
        {"action": "resume", "job_id": job.job_id, "resume_request_id": "again"}
    )
    snapshot = snapshot_receipt(job, state_no_sim)
    evaluation = JobsEvaluation(
        args=args,
        snapshot=snapshot,
        job_id=job.job_id,
        request_id=job.request_id,
        status=job.status,
        resumed=False,
        replayed=True,
        head_job_id=job.job_id,
    )
    reply = await jobs_module._present_jobs(args, evaluation, None, state_no_sim)
    assert "path" in reply["source"][0]


# ---------------------------------------------------------------------------
# 5. The deck's .meas, from what the runner kept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_meas_route_reads_what_the_runner_kept_and_no_deck(state_no_sim: SessionState):
    job = _job(state_no_sim, "exp_kept_meas", 3)
    for case in job.cases:
        case.measurement_facts = {"requested": ["vmax"]}
        case.staged_deck = state_no_sim.working_dir / "gone" / f"{case.case_id}.cir"
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id)
    assert "The deck has .meas cards" in data["hint"]
