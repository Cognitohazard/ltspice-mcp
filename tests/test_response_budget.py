"""Contracts for the caller-set response budget and its degradation ladder.

Three claims are worth a test each and are the reason this file exists: a
budget never costs the caller a fact, a budget never changes what a result IS,
and a page shrunk to fit a budget still pages to every row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import response_budget
from ltspice_mcp.lib.response_budget import Rung
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools as insp
from ltspice_mcp.tools.analyze import (
    OUTPUT_SCHEMA,
    AnalyzeResultsInput,
    _request_hash,
    handle_analyze_results,
)
from ltspice_mcp.tools.experiments import (
    JOBS_OUTPUT_SCHEMA,
    JobsInput,
    handle_jobs,
)
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests.conftest import make_batch_job, stage_recorded_fixture

# A .step AC sweep: 45 attributed rows off one raw, which is the shape a budget
# has anything to say about.
_WIDE_RECIPE: dict[str, Any] = {
    "key": "loop",
    "metric": "bode_filter",
    "signal": "V(out)",
    "all_steps": True,
}
_WIDE_SOURCES = 15


def _wide_args(raw: Path, **extra: Any) -> AnalyzeResultsInput:
    return AnalyzeResultsInput.model_validate(
        {
            "sources": [
                {"raw_path": str(raw), "label": f"corner{index:02d}"}
                for index in range(_WIDE_SOURCES)
            ],
            "recipes": [_WIDE_RECIPE],
            **extra,
        }
    )


async def _analysis(state: SessionState, raw: Path, **extra: Any) -> dict[str, Any]:
    result = await handle_analyze_results(_wide_args(raw, **extra), state)
    assert result.structuredContent is not None
    jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(result.structuredContent)
    return result.structuredContent


def _observation(data: dict[str, Any], code: str) -> dict[str, Any] | None:
    for item in data.get("observations", []):
        if item.get("code") == code:
            return item
    return None


def _stable(data: Any) -> str:
    """JSON of ``data`` with the per-call result-set identity blanked.

    A result set gets a fresh random id on every call, and every cursor embeds
    it, so two runs of the same request never share those bytes — with or
    without a budget. Blanking them is what makes an equality claim about the
    REST of the response meaningful.
    """
    text = json.dumps(data, sort_keys=True)
    if isinstance(data, dict) and isinstance(data.get("result_set_id"), str):
        text = text.replace(data["result_set_id"], "<result_set>")
    return text


def _uncolumnar(container: dict[str, Any], key: str) -> list[Any]:
    """The rows under ``key``, back in object form if the columnar rung fired."""
    rows = container[key]
    columns = container.get(response_budget.columnar_key(key))
    if columns is None:
        return rows
    return [dict(zip(columns, row, strict=True)) for row in rows]


# ---------------------------------------------------------------------------
# The ladder's primitives
# ---------------------------------------------------------------------------


class TestLadderPrimitives:
    def test_estimate_is_the_serializer_over_four(self):
        payload = {"a": [1, 2, 3], "b": "xyz"}
        expected = len(json.dumps(payload, separators=(",", ":"))) // 4
        assert response_budget.estimate_tokens(payload) == expected

    def test_columnar_is_lossless_and_reversible(self):
        rows = [{"a": 1, "b": None}, {"a": 2, "b": "x"}]
        block = {"items": [dict(row) for row in rows]}
        assert response_budget.columnarize(block, "items") is True
        assert block["items_columns"] == ["a", "b"]
        assert block["items"] == [[1, None], [2, "x"]]
        assert _uncolumnar(block, "items") == rows

    def test_columnar_refuses_rows_that_do_not_share_a_key_set(self):
        """Null-filling a missing column cannot be told apart from a real null,
        so a heterogeneous surface stays as it is rather than losing that."""
        block = {"items": [{"a": 1, "b": 2}, {"a": 2}]}
        assert response_budget.columnarize(block, "items") is False
        assert block["items"] == [{"a": 1, "b": 2}, {"a": 2}]
        assert response_budget.columnar_key("items") not in block

    def test_columnar_leaves_a_single_row_alone(self):
        block = {"items": [{"a": 1, "b": 2}]}
        assert response_budget.columnarize(block, "items") is False

    def test_fit_limit_never_grows_and_never_reaches_zero(self):
        rows = [{"a": "x" * 40} for _ in range(20)]
        generous = Rung(level=response_budget.RUNG_SHRINK, budget=1_000_000, measured=500)
        assert response_budget.fit_limit(20, rows, generous) == 20
        starved = Rung(level=response_budget.RUNG_SHRINK, budget=500, measured=100_000)
        assert response_budget.fit_limit(20, rows, starved) == 1

    def test_rung_zero_removes_optional_and_only_empties_required(self):
        container = {"optional": [], "kept": [1], "required": [1, 2]}
        response_budget.remove_when_empty(container, "optional")
        response_budget.remove_when_empty(container, "kept")
        response_budget.empty_required(container, "required")
        assert "optional" not in container
        assert container["kept"] == [1]
        assert container["required"] == []

    async def test_ladder_terminates_and_reports_a_budget_it_cannot_meet(self):
        """A budget under the floor gets the floor, not an endless descent."""
        rungs: list[int] = []

        async def render(rung: Rung) -> dict[str, Any]:
            rungs.append(rung.level)
            return {"failures": ["x" * 4000]}

        result = await response_budget.negotiate(1, render)
        assert rungs == list(response_budget.LADDER)
        assert result.met is False
        assert result.rung.level == response_budget.RUNG_SHRINK
        assert (
            "could not be met" in response_budget.not_met_observation(result.rung, 999)["detail"]
        )


# ---------------------------------------------------------------------------
# analyze_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnalysisBudget:
    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        """Absent and comfortably-met budgets return the same bytes: a budget
        is a ceiling, not a rendering mode."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw)
        met = await _analysis(state_no_sim, raw, budget=1_000_000)
        assert _stable(met) == _stable(plain)
        assert _observation(met, "budget_truncated") is None

    async def test_budget_is_not_part_of_a_request_identity(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The budget lives outside the hash a cursor is checked against, so two
        budgets over one analysis are the same request and share its cursors."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = _request_hash(_wide_args(raw))
        budgeted = _request_hash(_wide_args(raw, budget=600))
        assert budgeted == plain

    async def test_a_cursor_minted_under_a_budget_resumes_without_one(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The per_run cursor is checked against the request hash; a budget
        outside that hash is what lets these two calls be the same request."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = await _analysis(state_no_sim, raw, budget=900, include={"per_run": {"limit": 20}})
        cursor = first["results"]["loop"]["per_run"]["next_cursor"]
        assert cursor is not None
        resumed = await _analysis(
            state_no_sim, raw, include={"per_run": {"limit": 20, "cursor": cursor}}
        )
        assert resumed["results"]["loop"]["per_run"]["returned"] > 0

    async def test_facts_survive_the_smallest_budget(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A budget squeezes presentation. Failures, observations, coverage and
        the spec verdict come back whole or the response is a lie."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        recipes = [
            _WIDE_RECIPE,
            {"key": "absent", "metric": "bode_filter", "signal": "V(nope)"},
        ]
        args = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": recipes,
                "budget": response_budget.BUDGET_MIN_TOKENS,
            }
        )
        result = await handle_analyze_results(args, state_no_sim)
        assert result.structuredContent is not None
        data = result.structuredContent
        jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)
        assert data["failures"], "a failing recipe's record is a fact, not presentation"
        assert data["coverage"]["runs_requested"] == 1
        assert data["coverage"]["runs_analyzed"] == 1
        assert _observation(data, "budget_truncated") is not None
        # Required keys are emptied, never deleted — that is what keeps every
        # rung inside the declared schema.
        assert "source_hashes" in data
        assert "next" in data and "cursor" in data

    async def test_over_budget_by_honesty_is_stated(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A response whose fact floor is bigger than the budget comes back over
        it — and says so rather than dropping facts to fit."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        data = await _analysis(
            state_no_sim,
            raw,
            budget=response_budget.BUDGET_MIN_TOKENS,
            include={"per_run": {"limit": 45}},
        )
        floor = response_budget.estimate_tokens(data)
        observation = _observation(data, "budget_not_met")
        if floor > response_budget.BUDGET_MIN_TOKENS:
            assert observation is not None
            assert "over it by honesty" in observation["detail"]
        else:
            assert observation is None

    async def test_every_budget_in_the_walk_stays_inside_the_schema(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The ladder has no rung that can emit something this tool's own schema
        rejects — walked from a met budget down to the floor."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw, include={"per_run": {"limit": 45}})
        full = response_budget.estimate_tokens(plain)
        for divisor in (1, 2, 4, 8, 16, 64):
            budget = max(full // divisor, response_budget.BUDGET_MIN_TOKENS)
            data = await _analysis(
                state_no_sim, raw, budget=budget, include={"per_run": {"limit": 45}}
            )
            jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)

    async def test_the_deepest_rung_applies_every_rung_above_it(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Rungs are cumulative, so the floor response is where all four show at
        once: empty identity echo, revoked opt-ins, columnar rows, shrunk page."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}, "provenance": True, "signals_available": True}
        plain = await _analysis(state_no_sim, raw, include=include)
        assert plain["source_hashes"][0].get("raw_sha256") is not None
        assert "signals_available" in plain
        assert isinstance(plain["results"]["loop"]["per_run"]["items"][0], dict)

        data = await _analysis(
            state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS, include=include
        )
        detail = _observation(data, "budget_truncated")["detail"]  # type: ignore[index]
        assert "rung 3 (shrink)" in detail
        # rung 0: required identity echo emptied, never removed.
        assert data["source_hashes"] == []
        # rung 1: the payload-growing opt-ins are revoked.
        assert "signals_available" not in data
        # rung 3: the page shrank, and it shrank before the cursor was minted —
        # returned is what the page holds, and the cursor resumes after it.
        # (Rung 2 leaves a one-row page alone: a column list the size of the row
        # it describes saves nothing. The columnar rung is exercised where it can
        # fire, in test_columnar_rows_carry_the_same_values.)
        page = data["results"]["loop"]["per_run"]
        assert page["returned"] < plain["results"]["loop"]["per_run"]["returned"]
        assert page["returned"] == len(page["items"])
        assert page["next_cursor"] is not None

    async def test_shrunk_pages_still_reach_every_row(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The point of shrinking before assembly: a cursor from a shrunk page
        resumes at the row after the last one shown — no gap, no repeat."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw, include={"per_run": {"limit": 45}})
        page = plain["results"]["loop"]["per_run"]
        expected = [(row["source"], row["step_index"]) for row in page["items"]]
        assert page["total"] == len(expected) > 40

        seen: list[tuple[Any, Any]] = []
        cursor: str | None = None
        for _ in range(400):
            per_run: dict[str, Any] = {"limit": 45}
            if cursor is not None:
                per_run["cursor"] = cursor
            page_data = await _analysis(
                state_no_sim,
                raw,
                budget=response_budget.BUDGET_MIN_TOKENS,
                include={"per_run": per_run},
            )
            page = page_data["results"]["loop"]["per_run"]
            rows = _uncolumnar(page, "items")
            assert rows, "a shrunk page must still carry at least one row"
            assert page["returned"] < 45, "this budget is supposed to shrink the page"
            seen.extend((row["source"], row["step_index"]) for row in rows)
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert cursor is None, "the shrunk page chain never terminated"
        assert seen == expected

    async def test_the_continuation_cursor_agrees_with_the_shrunk_page(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A shrunk page mints TWO resume tokens — the per_run cursor and the
        call-level 'next'. Both have to point at the same next row, or the
        caller's choice of route decides how many rows it silently loses."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}}
        plain = await _analysis(state_no_sim, raw, include=include)
        expected = plain["results"]["loop"]["per_run"]["items"]

        first = await _analysis(
            state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS, include=include
        )
        shown = first["results"]["loop"]["per_run"]["returned"]
        assert 0 < shown < len(expected)
        assert first["next"] is not None

        resumed = await handle_analyze_results(
            AnalyzeResultsInput.model_validate({"continue": first["next"]}),
            state_no_sim,
        )
        assert resumed.structuredContent is not None
        rows = _uncolumnar(resumed.structuredContent["results"]["loop"]["per_run"], "items")
        assert rows[0] == expected[shown], "the continuation skipped rows the page never showed"

    async def test_the_same_budget_gives_the_same_bytes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = await _analysis(state_no_sim, raw, budget=800)
        second = await _analysis(state_no_sim, raw, budget=800)
        assert _stable(first) == _stable(second)

    async def test_columnar_rows_carry_the_same_values(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Columnar is a presentation change: the same rows, without the key
        names repeated on every one of them."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}}
        plain = await _analysis(state_no_sim, raw, include=include)
        expected = plain["results"]["loop"]["per_run"]["items"]
        full = response_budget.estimate_tokens(plain)

        columnar: dict[str, Any] | None = None
        for divisor in (2, 3, 4, 6, 8, 12, 16, 24, 32):
            budget = max(full // divisor, response_budget.BUDGET_MIN_TOKENS)
            data = await _analysis(state_no_sim, raw, budget=budget, include=include)
            page = data["results"]["loop"]["per_run"]
            if response_budget.columnar_key("items") in page:
                columnar = page
                break
        assert columnar is not None, "no budget in the walk reached the columnar rung"
        rows = _uncolumnar(columnar, "items")
        assert len(rows) > 1
        assert rows == expected[: len(rows)]

    async def test_a_budget_below_the_floor_is_refused_at_the_edge(self):
        """Not a silent clamp: a number the ladder could never negotiate is a
        request error, so the caller learns the floor instead of guessing."""
        with pytest.raises(Exception, match="greater than or equal to 500"):
            AnalyzeResultsInput.model_validate(
                {"sources": [{"raw_path": "x.raw"}], "recipes": [_WIDE_RECIPE], "budget": 1}
            )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _batch_with_runs(work_dir: Path, count: int):
    return make_batch_job(
        "b_budget",
        total_runs=count,
        completed_runs=count,
        run_results={
            index: {
                "raw_file": str(work_dir / f"run_{index:04d}.raw"),
                "log_file": str(work_dir / f"run_{index:04d}.log"),
                "params": {"R1": f"{index + 1}k", "C1": f"{index + 1}n"},
            }
            for index in range(count)
        },
    )


async def _jobs(state: SessionState, **values: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(values), state)
    assert result.structuredContent is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(result.structuredContent)
    return result.structuredContent


@pytest.mark.asyncio
class TestJobsBudget:
    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(work_dir, 60)
        state_no_sim.all_jobs[job.job_id] = job
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        met = await _jobs(state_no_sim, action="runs", job_id=job.job_id, budget=1_000_000)
        assert _stable(met) == _stable(plain)

    async def test_shrunk_run_pages_reach_every_record(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        job = _batch_with_runs(work_dir, 60)
        state_no_sim.all_jobs[job.job_id] = job
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        assert plain["total"] == 60
        expected = [f"{job.job_id}-case-{index:04d}" for index in range(60)]

        seen: list[str] = []
        cursor: str | None = None
        for _ in range(200):
            page = await _jobs(
                state_no_sim,
                action="runs",
                job_id=job.job_id,
                budget=response_budget.BUDGET_MIN_TOKENS,
                **({"cursor": cursor} if cursor is not None else {}),
            )
            rows = _uncolumnar(page, "items")
            assert rows
            assert page["returned"] < 50, "this budget is supposed to shrink the page"
            seen.extend(row["case_id"] for row in rows)
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert cursor is None
        assert seen == expected

    async def test_the_answer_rung_drops_artifact_paths_only_from_produced_runs(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A produced run's paths are provenance the analysis tools resolve by
        id; a run that did not produce keeps them, because its log IS the
        diagnostic and failures entries carry only a code and a message."""
        job = _batch_with_runs(work_dir, 60)
        job.run_results[7] = {"raw_file": None, "log_file": None, "params": {}}
        state_no_sim.all_jobs[job.job_id] = job
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        assert "raw" in plain["items"][0]

        page = await _jobs(
            state_no_sim,
            action="runs",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        rows = _uncolumnar(page, "items")
        assert all("raw" not in row for row in rows if row["status"] == "produced")
        assert all("raw" in row for row in rows if row["status"] != "produced")

    async def test_facts_and_handles_survive(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(work_dir, 60)
        state_no_sim.all_jobs[job.job_id] = job
        data = await _jobs(
            state_no_sim,
            action="runs",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert data["failures"] == []
        assert data["job_id"] == job.job_id
        assert data["status"] == job.status
        assert _observation(data, "budget_truncated") is not None


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _many_component_netlist(work_dir: Path, count: int) -> Path:
    path = work_dir / "budget_deck.cir"
    lines = ["* budget deck"]
    lines += [f"R{index} n{index} 0 {index + 1}k" for index in range(count)]
    lines += [".op", ".end"]
    path.write_text("\n".join(lines) + "\n")
    return path


async def _inspect(state: SessionState, queries: list[dict[str, Any]], **extra: Any):
    result = await handle_inspect(
        InspectInput.model_validate({"queries": queries, **extra}), state
    )
    assert result.structuredContent is not None
    return result.structuredContent


@pytest.mark.asyncio
class TestInspectBudget:
    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        path = _many_component_netlist(work_dir, 90)
        query = {"kind": "components", "path": str(path)}
        plain = await _inspect(state_no_sim, [query])
        met = await _inspect(state_no_sim, [query], budget=1_000_000)
        assert _stable(met) == _stable(plain)

    async def test_shrunk_pages_reach_every_component(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(insp, "_PAGE_SIZE", 250)
        path = _many_component_netlist(work_dir, 250)
        plain = await _inspect(state_no_sim, [{"kind": "components", "path": str(path)}])
        expected = plain["results"][0]["data"]["components"]
        assert len(expected) == 250

        seen: list[Any] = []
        cursor: str | None = None
        for _ in range(200):
            query: dict[str, Any] = {"kind": "components", "path": str(path)}
            if cursor is not None:
                query["cursor"] = cursor
            data = await _inspect(state_no_sim, [query], budget=response_budget.BUDGET_MIN_TOKENS)
            item = data["results"][0]
            assert item["ok"] is True
            payload = item["data"]
            rows = _uncolumnar(payload, "components")
            assert rows
            assert len(rows) < 250, "this budget is supposed to shrink the page"
            seen.extend(rows)
            cursor = item.get("next_cursor")
            if cursor is None:
                break
        assert cursor is None
        assert seen == expected

    async def test_a_failing_query_still_reports_its_error(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Per-item isolation is a fact channel: no budget hides why a query
        failed, and the batch outcome still says the batch was partial."""
        path = _many_component_netlist(work_dir, 90)
        data = await _inspect(
            state_no_sim,
            [
                {"kind": "components", "path": str(path)},
                {"kind": "components", "path": str(work_dir / "missing.cir")},
            ],
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert data["outcome"] == "partial"
        failed = data["results"][1]
        assert failed["ok"] is False
        assert failed["error"]["code"]
        assert failed["error"]["message"]
        assert _observation(data, "budget_truncated") is not None
        assert "budget" in data["hint"]

    async def test_the_answer_rung_falls_back_to_the_component_list(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """detail='full' is the one opt-in inspect has, so it is what the answer
        rung revokes — and the response says which detail it actually rendered."""
        path = _many_component_netlist(work_dir, 90)
        data = await _inspect(
            state_no_sim,
            [{"kind": "components", "path": str(path), "detail": "full"}],
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert data["results"][0]["data"]["detail"] == "list"
