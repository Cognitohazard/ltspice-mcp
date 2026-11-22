"""``runs_not_shown`` has one bound in tokens, and the size search keeps as
many run rows as fit beside it.

Design principle "Jobs and results are stored on disk and requests are repeatable": a reply that leaves
rows out names the facts on them in a part that is never cut, so that part is
bounded by itself, whatever the number of kinds and runs. The rows are then
sized against what that part leaves of the target.
"""

from __future__ import annotations

import random
import threading
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


@pytest.mark.asyncio
async def test_a_runs_page_reads_the_facts_of_every_row_off_the_loop(
    state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    """The facts of a job's whole table are read for every page: a job of
    many rows would hold up every other request if that ran on the loop."""
    job = _job(state_no_sim, "exp_runs_off_loop", 40)
    loop_thread = threading.get_ident()
    read_on: list[int] = []
    real = receipts.RowFacts.of.__func__

    def recording(cls: Any, rows: Any) -> Any:
        read_on.append(threading.get_ident())
        return real(cls, rows)

    monkeypatch.setattr(receipts.RowFacts, "of", classmethod(recording))
    data = await _jobs(state_no_sim, action="runs", job_id=job.job_id, budget=500)
    assert data["items"]
    assert read_on and loop_thread not in read_on


def test_a_band_of_counts_that_fit_below_the_largest_is_found(tmp_path):
    """The note shrinks as rows are kept, as ``runs_not_shown`` does: large
    below 60 rows, a middle size from 60 on. Then only a band in the middle
    fits whole (60 to 90 rows), below the counts that fit without the note
    and away from one row. The largest of that band is kept, and the reply
    does not say it is over its target."""
    rows = [f"r{index:04d}" for index in range(300)]
    huge = ["fact " + "z" * 40] * 300
    reserve = estimate_tokens({"left_out": huge})
    middle: list[str] = []

    def annotate(reply: dict[str, Any], cut: Any, _stored: Any, _paged: Any) -> None:
        entry = cut.get("/rows")
        if entry is None:
            return
        note = huge if entry.returned < 60 else middle
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

    def size(count: int) -> int:
        return estimate_tokens(presented(10**9, cap=count).reply, as_sent=False)

    target = size(250)  # no note above 60 yet: 250 rows fit without it
    while size(90) <= target:  # the middle note grows until 90 rows just fit
        middle.append("zzzz")
    middle.pop()
    fits = [count for count in range(1, 301) if size(count) <= target]
    assert fits[0] >= 60 and fits[-1] == 90, fits
    assert reserve > target and estimate_tokens({"rows": rows}, as_sent=False) > target

    shown = presented(target)
    assert shown.met
    assert len(shown.reply["rows"]) == 90


def _over(data: dict[str, Any]) -> bool:
    return any(item["code"] == "budget_not_met" for item in data.get("observations", []))


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", range(900, 4051, 150))
async def test_a_note_that_grows_with_the_rows_kept_does_not_send_a_reply_over(
    state_no_sim: SessionState, budget: int
):
    """The second run carries two codes and the third a third, so the note at
    one row names two runs; from run 40 on the three fall on separate runs,
    and the note at a larger count names three. The reserve, measured at one
    row, is below that note: a reply is over its target only at one row."""
    job = _job(state_no_sim, f"exp_grows_{budget}", 300)
    job.cases[1].log_diagnostics = _diagnosed(_CODES[0:2])
    job.cases[2].log_diagnostics = _diagnosed(_CODES[2:3])
    for index in range(40, 300):
        job.cases[index].log_diagnostics = _diagnosed([_CODES[index % 3]])
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=budget)
    if _over(data):
        assert len(data["runs"]["items"]) == 1, (budget, len(data["runs"]["items"]))
    else:
        assert estimate_tokens(data) <= budget


@pytest.mark.parametrize("seed", range(60))
def test_no_reply_is_over_its_target_while_a_smaller_count_fits(tmp_path, seed: int):
    """Facts placed at random, one to three kinds a run, and a note shaped as
    ``runs_not_shown`` is: every kind's count, and up to three runs named,
    each the first left out with a kind not yet named and at most two of its
    kinds. A run with several kinds early and runs with one kind later make
    the note grow as rows are kept."""
    rng = random.Random(seed)
    count = rng.randint(20, 150)
    rows = [f"row {index:04d} " + "y" * rng.randint(4, 40) for index in range(count)]
    kinds = [f"kind{number}" for number in range(6)]
    facts = {
        index: rng.sample(kinds, rng.randint(1, 3))
        for index in rng.sample(range(count), rng.randint(3, min(count, 30)))
    }

    def note_from(start: int) -> dict[str, Any] | None:
        left = {index: carried for index, carried in facts.items() if index >= start}
        if not left:
            return None
        counts: dict[str, int] = {}
        for carried in left.values():
            for kind in carried:
                counts[kind] = counts.get(kind, 0) + 1
        named: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index in sorted(left):
            fresh = [kind for kind in left[index] if kind not in seen][:2]
            if fresh and len(named) < 3:
                seen.update(fresh)
                named.append({"run": f"run {index:04d} " + "z" * 30, "kinds": fresh})
        return {"counts": counts, "runs": named}

    def annotate(reply: dict[str, Any], cut: Any, _stored: Any, _paged: Any) -> None:
        entry = cut.get("/rows")
        note = note_from(0 if entry is None else entry.returned)
        if entry is not None and note:
            reply["left_out"] = note

    first = note_from(1)
    reserve = estimate_tokens({"left_out": first}) if first else 0

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
            apart={"left_out": reserve} if reserve else None,
        )

    target = rng.randint(80, max(81, estimate_tokens({"rows": rows}, as_sent=False)))
    shown = presented(target)
    kept = len(shown.reply["rows"])
    if not shown.met and kept > 1:
        smaller = [
            c
            for c in range(1, kept)
            if estimate_tokens(presented(10**9, cap=c).reply, as_sent=False) <= target
        ]
        assert not smaller, f"{kept} rows over {target} while {smaller[-1]} fit"


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


# ---------------------------------------------------------------------------
# 5. What a run's case observed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["saves_not_stored", "raw_names_disagree", "subcircuit_nodes_not_saved"]
)
async def test_a_case_observation_on_a_run_left_out_is_on_the_reply(
    state_no_sim: SessionState, code: str
):
    """A run's case observations are lifted to the reply's ``observations``,
    which is never cut, whatever rows the reply holds: a run left out keeps
    them, the same for every code a case can carry."""
    job = _job(state_no_sim, f"exp_case_obs_{code}", 300)
    job.cases[250].observations.append(
        {"code": code, "kind": "coverage", "detail": f"{code} on run 250"}
    )
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=600)
    assert len(data["runs"]["items"]) < 250
    assert [item for item in data["observations"] if item["code"] == code]
