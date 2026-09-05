"""Contracts for the caller-set response budget and its degradation ladder.

Three claims are worth a test each and are the reason this file exists: a
budget never costs the caller a fact, a budget never changes what a result IS,
and a page shrunk to fit a budget still pages to every row.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import response_budget
from ltspice_mcp.lib.response_budget import Rung
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import experiments as exp_mod
from ltspice_mcp.tools import inspect_tools as insp
from ltspice_mcp.tools import jobs as jobs_mod
from ltspice_mcp.tools import receipts as receipts_mod
from ltspice_mcp.tools._base import ResponseBudget
from ltspice_mcp.tools.analyze import (
    OUTPUT_SCHEMA,
    AnalyzeResultsInput,
    _request_hash,
    handle_analyze_results,
)
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.jobs import (
    JOBS_OUTPUT_SCHEMA,
    JobsInput,
    handle_jobs,
)
from tests.conftest import SyncApi, make_experiment_job, stage_recorded_fixture

# Every rung-0 allowlist the three budget-aware tools declare, paired with the
# schema node whose keys it names. Listed rather than derived: the coverage test
# below fails on any `_TRIM_*` constant that is not here, so a new allowlist
# cannot slip in unpinned.
_TRIM_ALLOWLISTS: list[tuple[Any, str, dict[str, Any]]] = [
    (analyze_mod, "_TRIM_REMOVE_RESULT", analyze_mod._RESULT_ENTRY_SCHEMA),
    (analyze_mod, "_TRIM_REMOVE_ENVELOPE", OUTPUT_SCHEMA),
    (analyze_mod, "_TRIM_EMPTY_ENVELOPE", OUTPUT_SCHEMA),
    (receipts_mod, "_TRIM_REMOVE_RECEIPT", receipts_mod.RUN_EXPERIMENTS_OUTPUT_SCHEMA),
    (receipts_mod, "_TRIM_REMOVE_RECEIPT", jobs_mod._jobs_receipt_schema("status")),
    (insp, "_TRIM_REMOVE_EXHAUSTED", insp._OUTPUT_SCHEMA["properties"]["results"]["items"]),
]


class TestRungZeroAllowlists:
    """Rung 0 against the schemas it edits.

    It is the one rung that exempts content, so its keys are declared as data
    and checked here rather than reasoned about at the call site. A ``REMOVE``
    key must be optional — deleting a required key would make the emission fail
    the tool's own schema — and an ``EMPTY`` key must be required, because an
    optional key that is only ever emptied should have been removed instead.
    """

    @pytest.mark.parametrize(
        ("module", "name", "schema"),
        _TRIM_ALLOWLISTS,
        ids=[
            f"{module.__name__.rsplit('.', 1)[-1]}.{name}" for module, name, _ in _TRIM_ALLOWLISTS
        ],
    )
    def test_key_classification_matches_the_schema(self, module, name, schema):
        keys = getattr(module, name)
        assert keys, f"{name} is empty; delete it rather than declaring a no-op allowlist"
        required = set(schema.get("required", ()))
        for key in keys:
            assert key in schema["properties"], f"{name} names {key!r}, absent from the schema"
            if "_REMOVE_" in name:
                assert key not in required, f"{name} would delete required key {key!r}"
            else:
                assert key in required, f"{name} empties optional key {key!r}; remove it instead"

    def test_every_declared_allowlist_is_pinned(self):
        """The fail-closed half: a rung-0 list added to a tool and not listed
        above is an exemption nothing checks."""
        for module in (analyze_mod, exp_mod, insp):
            declared = {name for name in vars(module) if name.startswith("_TRIM_")}
            pinned = {name for mod, name, _ in _TRIM_ALLOWLISTS if mod is module}
            assert declared == pinned, f"{module.__name__}: unpinned rung-0 allowlists"


# A .step AC sweep: 45 attributed rows off one raw, which is the shape a budget
# has anything to say about.
_WIDE_RECIPE: dict[str, Any] = {
    "key": "loop",
    "metric": "bode_filter",
    "signal": "V(out)",
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
            "all_steps": True,
            **extra,
        }
    )


async def _analysis(state: SessionState, raw: Path, **extra: Any) -> dict[str, Any]:
    result = await handle_analyze_results(_wide_args(raw, **extra), state)
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


# The same fan-out with a spec every sample fails, so spec.fail_cases — not the
# per-run page — is what the response is mostly made of.
_SPEC_RECIPE: dict[str, Any] = {
    "key": "vout",
    "metric": "value",
    "expr": "V(out)",
    "at": "900u",
    "spec": {"max": -1.0},
}


async def _spec_analysis(state: SessionState, raw: Path, **extra: Any) -> dict[str, Any]:
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [
                    {"raw_path": str(raw), "label": f"corner{index:02d}"}
                    for index in range(_WIDE_SOURCES)
                ],
                "recipes": [_SPEC_RECIPE],
                "all_steps": True,
                **extra,
            }
        ),
        state,
    )
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


# Fractions of an undegraded response's own size, spanning a met budget down
# past the floor. A rung that only misbehaves partway down the ladder is
# invisible to a floor-only probe, so every walk in this file covers the spread.
_WALK_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64)


def _budget_walk(full: int) -> list[int]:
    """The distinct budgets to try against a response measuring ``full`` tokens.

    Distinct because the deeper divisors all clamp to the floor, and the same
    budget twice returns the same bytes — paying for that twice buys nothing.
    """
    return sorted(
        {max(full // divisor, response_budget.BUDGET_MIN_TOKENS) for divisor in _WALK_DIVISORS},
        reverse=True,
    )


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


# ---------------------------------------------------------------------------
# The ladder's primitives
# ---------------------------------------------------------------------------


class TestLadderPrimitives:
    def test_estimate_is_the_serializer_over_four(self):
        payload = {"a": [1, 2, 3], "b": "xyz"}
        expected = len(json.dumps(payload, separators=(",", ":"))) // 4
        assert response_budget.estimate_tokens(payload) == expected

    def test_fit_limit_never_grows_and_never_reaches_zero(self):
        measure = response_budget.RowMeasure.of([{"a": "x" * 40} for _ in range(20)])
        generous = Rung(level=response_budget.RUNG_SHRINK, budget=1_000_000, measured=500)
        assert measure.fit_limit(20, generous) == 20
        starved = Rung(level=response_budget.RUNG_SHRINK, budget=500, measured=100_000)
        assert measure.fit_limit(20, starved) == 1

    def test_rung_zero_removes_optional_and_only_empties_required(self):
        container = {"optional": [], "kept": [1], "required": [1, 2]}
        response_budget.apply_trim(container, remove=("optional", "kept"), empty=("required",))
        assert "optional" not in container
        assert container["kept"] == [1]
        assert container["required"] == []

    def test_notes_extend_observations_rather_than_replacing_them(self):
        """A budget cuts presentation, so it may never overwrite a fact channel
        a tool already filled — the epilogue appends, on every tool."""
        rung = Rung(level=response_budget.RUNG_SHRINK, budget=600, measured=99_999)
        result = response_budget.Negotiated(
            data={"observations": [{"code": "prior"}], "hint": "keep me"},
            rung=rung,
            estimate=99_999,
        )
        notes = response_budget.Notes(cut="cut", route="route", hint_key="hint")
        response_budget.attach_notes(result, notes)
        codes = [o["code"] for o in result.data["observations"]]
        assert codes == ["prior", "budget_truncated", "budget_not_met"]
        assert result.data["hint"].startswith("keep me ")

    def test_append_hint_preserves_the_route_and_deduplicates_detail(self):
        data = {"hint": "keep me"}
        response_budget.append_hint(data, "continue here")
        response_budget.append_hint(data, "continue here")

        assert data["hint"] == "keep me continue here"

    async def test_ladder_terminates_and_reports_a_budget_it_cannot_meet(self):
        """A budget under the floor gets the floor, not an endless descent."""
        rungs: list[int] = []

        async def render(rung: Rung) -> dict[str, Any]:
            rungs.append(rung.level)
            return {"failures": ["x" * 4000]}

        result = await response_budget.negotiate(
            1, render, response_budget.Notes(cut="cut", route="route")
        )
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
        is a ceiling, not a rendering mode.

        The server's own default is switched off here so 'absent' means what it
        says — with it on, a large default response is trimmed at rung 0 and the
        two are legitimately different (TestServerDefaultBudget covers that)."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 0
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

    async def test_fields_view_is_not_part_of_per_run_request_identity(self, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = _request_hash(
            _wide_args(raw, include={"per_run": {"limit": 5}, "fields": ["value"]})
        )
        second = _request_hash(
            _wide_args(raw, include={"per_run": {"limit": 5}, "fields": ["case_id"]})
        )
        assert first == second

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
        assert result.structured_content is not None
        data = result.structured_content
        jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)
        assert data["failures"], "a failing recipe's record is a fact, not presentation"
        assert data["coverage"]["runs_requested"] == 1
        assert data["coverage"]["runs_analyzed"] == 1
        assert _observation(data, "budget_truncated") is not None
        # Required keys are emptied, never deleted — that is what keeps every
        # rung inside the declared schema.
        assert "source_hashes" in data
        assert "next" in data and "cursor" in data

    async def test_a_response_over_its_budget_says_so(
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
            assert "over the budget instead" in observation["detail"]
        else:
            assert observation is None

    async def test_every_budget_in_the_walk_stays_inside_the_schema(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The ladder has no rung that can emit something this tool's own schema
        rejects — walked from a met budget down to the floor."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw, include={"per_run": {"limit": 45}})
        for budget in _budget_walk(response_budget.estimate_tokens(plain)):
            data = await _analysis(
                state_no_sim, raw, budget=budget, include={"per_run": {"limit": 45}}
            )
            jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)

    async def test_the_deepest_rung_applies_every_rung_above_it(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Rungs are cumulative, so the floor response is where all three show
        at once: empty identity echo, revoked opt-ins, shrunk page."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}, "provenance": True, "signals_available": True}
        # The undegraded reference this test measures the rungs against; the
        # server's own default would already have applied rung 0 to it.
        state_no_sim.config.default_budget = 0
        plain = await _analysis(state_no_sim, raw, include=include)
        assert plain["source_hashes"][0].get("raw_sha256") is not None
        assert "signals_available" in plain
        assert isinstance(plain["results"]["loop"]["per_run"]["items"][0], dict)

        data = await _analysis(
            state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS, include=include
        )
        detail = _observation(data, "budget_truncated")["detail"]  # type: ignore[index]
        assert "rung 2 (shrink)" in detail
        # rung 0: required identity echo emptied, never removed.
        assert data["source_hashes"] == []
        # rung 1: the payload-growing opt-ins are revoked.
        assert "signals_available" not in data
        # rung 2: the page shrank, and it shrank before the cursor was minted —
        # returned is what the page holds, and the cursor resumes after it.
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
            rows = page["items"]
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
        assert resumed.structured_content is not None
        rows = resumed.structured_content["results"]["loop"]["per_run"]["items"]
        assert rows[0] == expected[shown], "the continuation skipped rows the page never showed"

    async def test_the_same_budget_gives_the_same_bytes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = await _analysis(state_no_sim, raw, budget=800)
        second = await _analysis(state_no_sim, raw, budget=800)
        assert _stable(first) == _stable(second)

    async def test_a_spec_heavy_call_shrinks_its_fail_cases(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """spec.fail_cases used to sit at a fixed cap the ladder never touched,
        so a call whose size driver was its failing cases had nothing to give and
        degraded to budget_not_met. The page shrinks; the fail COUNT does not."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        plain = await _spec_analysis(state_no_sim, raw)
        wide = plain["results"]["vout"]["spec"]
        assert wide["fail_count"] > 4, "fixture stopped producing failing cases"
        assert wide["fail_cases"]["truncated"] is False

        data = await _spec_analysis(state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS)
        spec = data["results"]["vout"]["spec"]
        page = spec["fail_cases"]["items"]
        assert len(page) < wide["fail_cases"]["returned"]
        assert spec["fail_count"] == wide["fail_count"], "a count is a fact, not a page"
        assert spec["verdict"] == wide["verdict"]

    async def test_a_grouped_call_shrinks_its_groups_and_says_so(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A group list is an aggregate, not a page, so no cursor walks it — but
        the shrink rung must still be able to reach it, and must state what it
        dropped rather than hand back a silently shorter answer."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw, group_by=["r"])
        every_group = plain["results"]["loop"]["groups"]
        assert len(every_group) > 1

        data = await _analysis(
            state_no_sim, raw, group_by=["r"], budget=response_budget.BUDGET_MIN_TOKENS
        )
        entry = data["results"]["loop"]
        assert len(entry["groups"]) < len(every_group)
        assert any("group(s) omitted" in warning for warning in entry["warnings"])

    async def test_the_row_measurement_covers_every_surface_the_ladder_shrinks(self):
        """The cost model's two halves have to name the same surfaces. One the
        ladder shrinks but the measurement omits is billed to the fixed envelope,
        so the shrink rung sizes its page against a cost that is not real."""
        data = {
            "results": {
                "k": {
                    "reduced": ["reduced-row"],
                    "values": ["value-row"],
                    "groups": ["group-row"],
                    "per_run": {"items": ["per-run-row"]},
                    "spec": {"fail_cases": {"items": ["fail-row"]}},
                }
            },
            "coverage": {"missing_cases": {"items": ["missing-row"]}},
        }
        assert sorted(analyze_mod.analysis_rows(data)) == [
            "fail-row",
            "group-row",
            "missing-row",
            "per-run-row",
            "reduced-row",
            "value-row",
        ]

    async def test_a_budget_below_the_floor_is_refused_at_the_edge(self):
        """Not a silent clamp: a number the ladder could never negotiate is a
        request error, so the caller learns the floor instead of guessing."""
        with pytest.raises(Exception, match="greater than or equal to 500"):
            AnalyzeResultsInput.model_validate(
                {"sources": [{"raw_path": "x.raw"}], "recipes": [_WIDE_RECIPE], "budget": 1}
            )


def _run_rows(count: int) -> list[dict[str, Any]]:
    """``count`` produced run records — the row surface a receipt budget shrinks."""
    return [
        {
            "case_id": f"case-{index:03d}",
            "run_index": index,
            "circuit": "dut",
            "assignments": {"R1": f"{index + 1}k"},
            "status": "produced",
            "raw": f"/tmp/run-{index:03d}.raw",
            "log": f"/tmp/run-{index:03d}.log",
        }
        for index in range(count)
    ]


def _run_receipt_build(rows: list[dict[str, Any]]) -> receipts_mod.ReceiptBuild:
    """A receipt builder over ``rows``, paging to whatever limit a rung asks for."""

    def build(limit: int, _rung: Rung | None) -> receipts_mod.ReceiptBuilt:
        data = exp_mod._empty_payload("budgeted-runs")
        selected = rows[:limit]
        data.update(
            {
                "status": "completed",
                "outcome": "complete",
                "runs": {
                    "items": selected,
                    "total": len(rows),
                    "returned": len(selected),
                    "truncated": len(selected) < len(rows),
                    "next_cursor": f"o:{len(selected)}" if len(selected) < len(rows) else None,
                },
            }
        )
        return receipts_mod.finalize_receipt(data), "completed"

    return build


@pytest.mark.asyncio
async def test_run_receipt_shrink_cursor_starts_after_the_selected_candidate():
    rows = _run_rows(80)

    result = await receipts_mod.render_run_receipt(
        ResponseBudget(response_budget.BUDGET_MIN_TOKENS),
        _run_receipt_build(rows),
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(receipts_mod.RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(data)
    page = data["runs"]
    assert 0 < page["returned"] < 50
    assert page["next_cursor"] is not None
    offset = jobs_mod._decode_jobs_cursor(page["next_cursor"])
    assert offset == page["returned"]
    rendered_rows = page["items"]
    assert all("raw" not in row and "log" not in row for row in rendered_rows)
    assert [item["case_id"] for item in rendered_rows] == [
        item["case_id"] for item in rows[:offset]
    ]
    assert rows[offset]["case_id"] == f"case-{offset:03d}"


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _batch_with_runs(state, count: int):
    """A completed experiment with ``count`` produced cases — the runs page
    the jobs budget has to shrink."""
    return make_experiment_job(state, job_id="b_budget", count=count, status="completed")


async def _jobs(state: SessionState, **values: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(values), state)
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


@pytest.mark.asyncio
class TestServerDefaultBudget:
    """The budget that runs when the caller sets none.

    The ladder used to be entirely opt-in, and opting in requires already
    knowing the response is too big — which a caller learns by receiving it. The
    server therefore applies its own budget, but only at rung 0: that rung
    removes empty presentation blocks and the identity echo, and nothing else,
    so it can cut no fact and revoke no detail the caller asked for.
    """

    async def test_a_large_default_response_loses_the_identity_echo_only(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 500
        trimmed = await _analysis(state_no_sim, raw)

        # Rung 0's own allowlist, and the whole of what the default may do: the
        # identity echo emptied, the answer untouched.
        assert trimmed["source_hashes"] == []
        assert trimmed["results"]["loop"]["values"]
        # No note claiming a budget the caller never set went unmet — the ladder
        # stopped at rung 0 by policy, which is the policy working.
        assert _observation(trimmed, "budget_not_met") is None

    async def test_the_default_stops_at_rung_zero_where_a_caller_budget_would_not(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Same tiny number, two sources of authority, two different ladders."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 500
        defaulted = await _analysis(state_no_sim, raw, include={"per_run": {"limit": 20}})
        state_no_sim.config.default_budget = 0
        explicit = await _analysis(
            state_no_sim, raw, budget=500, include={"per_run": {"limit": 20}}
        )

        # The explicit budget walks past rung 0 and shrinks what the caller
        # asked for; the server's default leaves the requested page standing.
        assert defaulted["results"]["loop"]["per_run"]["returned"] == 20
        assert explicit["results"]["loop"]["per_run"]["returned"] < 20
        assert _observation(explicit, "budget_truncated") is not None
        # The default degraded too — at rung 0 — and says so. What it must not
        # say is "re-ask without 'budget'" to a caller who never passed one.
        defaulted_note = _observation(defaulted, "budget_truncated")
        assert defaulted_note is not None
        assert "rung 0" in defaulted_note["detail"]
        assert "larger 'budget'" in defaulted_note["detail"]

    async def test_zero_disables_the_default_entirely(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 0
        undegraded = await _analysis(state_no_sim, raw, include={"provenance": True})
        assert undegraded["source_hashes"], "nothing should have been trimmed"

    async def test_facts_come_back_whole_under_the_default(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 500
        args = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [
                    _WIDE_RECIPE,
                    {"key": "absent", "metric": "bode_filter", "signal": "V(nope)"},
                ],
            }
        )
        result = await handle_analyze_results(args, state_no_sim)
        data = result.structured_content
        assert data is not None
        assert data["failures"], "a failing recipe is a fact, not presentation"
        assert data["coverage"]["runs_analyzed"] >= 1


@pytest.mark.asyncio
class TestJobsBudget:
    async def test_answer_rung_rebuilds_a_receipt_at_the_same_page_limit(self):
        built_for: list[bool] = []

        def build(_limit: int, rung: Rung | None):
            answer_channel = rung is not None and rung.answer_channel
            built_for.append(answer_channel)
            return (
                {
                    "analysis": {
                        "view": "answer" if answer_channel else "detail",
                        "payload": "" if answer_channel else "x" * 4_000,
                    },
                    "observations": [],
                    "warnings": [],
                    "failures": [],
                },
                "receipt",
            )

        data, _ = await jobs_mod._negotiate_jobs(ResponseBudget(600), build, 50)

        assert built_for == [False, True]
        assert data["analysis"]["view"] == "answer"

    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(state_no_sim, 60)
        # As above: 'no budget' has to mean no budget for this comparison.
        state_no_sim.config.default_budget = 0
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        met = await _jobs(state_no_sim, action="runs", job_id=job.job_id, budget=1_000_000)
        assert _stable(met) == _stable(plain)

    async def test_shrunk_run_pages_reach_every_record(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        job = _batch_with_runs(state_no_sim, 60)
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        assert plain["total"] == 60
        expected = [f"case-{index:04d}" for index in range(60)]

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
            rows = page["items"]
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
        job = _batch_with_runs(state_no_sim, 60)
        job.cases[7].status = "failed"
        job.cases[7].raw_file = None
        job.cases[7].log_file = None
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        assert "raw" in plain["items"][0]

        page = await _jobs(
            state_no_sim,
            action="runs",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        rows = page["items"]
        assert all("raw" not in row for row in rows if row["status"] == "produced")
        assert all("raw" in row for row in rows if row["status"] != "produced")

    async def test_facts_and_handles_survive(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(state_no_sim, 60)
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
    assert result.structured_content is not None
    return result.structured_content


def test_the_api_automatic_door_gets_no_server_default(state_no_sim: SessionState, work_dir: Path):
    """The interface that promises complete results runs no presentation ladder.

    It refuses ``budget`` outright, so a response degraded there would route the
    caller at the one field that interface rejects — and its promise of complete
    results would be false while the server quietly trimmed the presentation.

    Sync rather than async because the interface's own bridge runs the call to
    completion on its own loop, which is the thing under test.
    """
    path = _many_component_netlist(work_dir, 90)
    query = {"kind": "components", "path": str(path), "detail": "full"}
    api = SyncApi(state_no_sim)

    state_no_sim.config.default_budget = 500
    defaulted = api.inspect(queries=[copy.deepcopy(query)])
    state_no_sim.config.default_budget = 0
    disabled = api.inspect(queries=[copy.deepcopy(query)])

    assert _stable(defaulted) == _stable(disabled), "the default trimmed the automatic mode"

    # Same session, same query, MCP: there the default does engage, so
    # it is the interface and not the configuration that decides.
    state_no_sim.config.default_budget = 500
    wire = asyncio.run(_inspect(state_no_sim, [copy.deepcopy(query)]))
    assert _observation(wire, "budget_truncated") is not None


def test_the_api_automatic_door_carries_no_budget_route(
    state_no_sim: SessionState, work_dir: Path
):
    """The note's route is "ask again with a larger 'budget'" — a field this
    API refuses. Checked on jobs(list), the collected surface that keeps the
    observations its pages carried."""
    for index in range(6):
        job = _batch_with_runs(state_no_sim, 20)
        job.job_id = f"b_budget_{index}"
        state_no_sim.all_jobs[job.job_id] = job
    state_no_sim.config.default_budget = 100

    complete = SyncApi(state_no_sim).jobs(action="list")
    wire = asyncio.run(_jobs(state_no_sim, action="list"))

    assert _observation(wire, "budget_truncated") is not None
    assert _observation(complete, "budget_truncated") is None


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
            rows = payload["components"]
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

    async def test_no_budget_in_a_sweep_lands_over_cap_without_saying_so(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Swept rather than spot-checked, because the failure was a band: the
        note this tool mirrors into 'hint' is written twice, and reserving room
        for one copy let responses across a whole stretch of budgets exceed the
        cap while flagged only as truncated. Over the cap is allowed — the fact
        floor is never cut to fit — but only when the response SAYS it is."""
        path = _many_component_netlist(work_dir, 90)
        query = {"kind": "components", "path": str(path), "detail": "full"}
        over_and_silent: list[tuple[int, int]] = []
        for budget in range(response_budget.BUDGET_MIN_TOKENS, 1400, 20):
            data = await _inspect(state_no_sim, [query], budget=budget)
            estimate = response_budget.estimate_tokens(data)
            if estimate > budget and _observation(data, "budget_not_met") is None:
                over_and_silent.append((budget, estimate))
        assert not over_and_silent, f"over cap, flagged only truncated: {over_and_silent}"

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


# ---------------------------------------------------------------------------
# Rows keep their shape at every budget
# ---------------------------------------------------------------------------


def _columns_siblings(node: Any, path: str = "") -> list[str]:
    """Every ``*_columns`` key in a response, by path.

    A positional row rendering cannot exist without one: arrays of values are
    unreadable unless something names the positions. So an empty list here is
    the machine-checkable form of "these rows are still objects", wherever in
    the envelope they sit.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key.endswith("_columns"):
                found.append(here)
            found.extend(_columns_siblings(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_columns_siblings(value, f"{path}[{index}]"))
    return found


def _assert_object_rows(rows: list[Any], within: set[str], *, where: str) -> None:
    """``rows`` are objects, keyed by names the untightened response also had."""
    assert rows, f"{where}: a budgeted page must still carry at least one row"
    for row in rows:
        assert isinstance(row, dict), f"{where}: row rendered positionally as {row!r}"
        assert set(row) <= within, f"{where}: row grew keys the plain response never had"


@pytest.mark.asyncio
class TestRowsKeepTheirShape:
    """A row is an object at every rung of the ladder.

    The ladder once carried a rung that re-rendered row surfaces positionally —
    a column list plus arrays of values — so a row's shape depended on how tight
    the budget was. That made the caller branch on the shape of what came back,
    so it was removed before 0.6.0. A smaller response now comes from a smaller
    page, which pages on through the same cursors.
    """

    async def test_analysis_rows_stay_objects_at_every_budget(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Swept rather than spot-checked: the deepest rung shrinks the page to
        a row or two, and it was the budgets ABOVE the floor that rendered a
        wide page positionally."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}}
        plain = await _analysis(state_no_sim, raw, include=include)
        keys = set(plain["results"]["loop"]["per_run"]["items"][0])

        for budget in _budget_walk(response_budget.estimate_tokens(plain)):
            data = await _analysis(state_no_sim, raw, budget=budget, include=include)
            assert _columns_siblings(data) == [], f"budget={budget} renamed its rows"
            _assert_object_rows(
                data["results"]["loop"]["per_run"]["items"], keys, where=f"budget={budget}"
            )

    async def test_spec_rows_stay_objects_at_every_budget(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The row surface beside per_run — a spec's failing cases — keeps its
        shape too, at every budget the ladder can be handed."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        plain = await _spec_analysis(state_no_sim, raw)
        keys = set(plain["results"]["vout"]["spec"]["fail_cases"]["items"][0])

        for budget in _budget_walk(response_budget.estimate_tokens(plain)):
            data = await _spec_analysis(state_no_sim, raw, budget=budget)
            assert _columns_siblings(data) == [], f"budget={budget} renamed its rows"
            _assert_object_rows(
                data["results"]["vout"]["spec"]["fail_cases"]["items"],
                keys,
                where=f"fail_cases budget={budget}",
            )

    async def test_run_receipt_rows_stay_objects(self):
        rows = _run_rows(80)
        result = await receipts_mod.render_run_receipt(
            ResponseBudget(response_budget.BUDGET_MIN_TOKENS),
            _run_receipt_build(rows),
        )
        data = result.structured_content
        assert data is not None
        jsonschema.Draft202012Validator(receipts_mod.RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(data)
        assert _columns_siblings(data) == []
        _assert_object_rows(data["runs"]["items"], set(rows[0]), where="receipt runs")

    async def test_jobs_run_page_rows_stay_objects(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        job = _batch_with_runs(state_no_sim, 60)
        plain = await _jobs(state_no_sim, action="runs", job_id=job.job_id)
        keys = set(plain["items"][0])

        data = await _jobs(
            state_no_sim,
            action="runs",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert _columns_siblings(data) == []
        _assert_object_rows(data["items"], keys, where="jobs runs")

    async def test_inspect_rows_stay_objects(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(insp, "_PAGE_SIZE", 250)
        path = _many_component_netlist(work_dir, 250)
        query = {"kind": "components", "path": str(path)}
        plain = await _inspect(state_no_sim, [query])
        keys = set(plain["results"][0]["data"]["components"][0])

        data = await _inspect(state_no_sim, [query], budget=response_budget.BUDGET_MIN_TOKENS)
        assert _columns_siblings(data) == []
        _assert_object_rows(
            data["results"][0]["data"]["components"], keys, where="inspect components"
        )


async def test_a_budget_shrinks_a_reference_lookup_instead_of_giving_up(
    state_no_sim: SessionState,
):
    """The reference kind reads no file, so it has no page and no cursor — but
    it does have a caller-visible size lever in ``limit``. The shrink rung
    reaches it, so a tight budget returns fewer branches rather than reporting
    a floor it could not meet, and the match count still says how many there
    were."""
    query = {"kind": "reference", "query": "gain", "limit": insp.REFERENCE_LIMIT_CAP}
    plain = await _inspect(state_no_sim, [query])
    tight = await _inspect(state_no_sim, [query], budget=response_budget.BUDGET_MIN_TOKENS)

    full_matches = plain["results"][0]["data"]["matches"]
    cut_matches = tight["results"][0]["data"]["matches"]
    assert len(full_matches) > 1
    assert len(cut_matches) < len(full_matches)
    # The number that matched is a fact and is not cut with the rows.
    assert tight["results"][0]["data"]["total_matches"] == len(full_matches)
    # Best-first ordering survives the shrink: what is dropped is the tail.
    assert [m["name"] for m in cut_matches] == [
        m["name"] for m in full_matches[: len(cut_matches)]
    ]
