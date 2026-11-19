"""Contracts for the caller-set response budget and its degradation ladder.

Four claims are worth a test each and are the reason this file exists: a
budget never costs the caller a fact, a budget never changes what a result IS,
a page shrunk to fit a budget still pages to every row, and the most degraded
experiment receipt costs the same however many cases the job ran.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import response_budget
from ltspice_mcp.lib.response_budget import Rung
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import inspect_tools as insp
from ltspice_mcp.tools import receipts as receipts_mod
from ltspice_mcp.tools.analyze import (
    OUTPUT_SCHEMA,
    AnalyzeResultsInput,
    _request_digest,
    handle_analyze_results,
)
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.jobs import (
    JOBS_OUTPUT_SCHEMA,
    JobsInput,
    handle_jobs,
)
from tests.conftest import (
    LIVENESS_S,
    SyncApi,
    make_experiment_job,
    recorded_fixture_simulator,
    stage_recorded_fixture,
    submit_experiment,
)
from tests.test_analyze_results import _analyze

# Every rung-0 allowlist the package declares, paired with the schema node whose
# keys it names. Listed rather than derived, because the pairing is the point:
# the coverage test below fails on any `_TRIM_*` constant declared anywhere in
# the package that is not here, so a new allowlist cannot slip in unpinned.
_TRIM_ALLOWLISTS: list[tuple[Any, str, dict[str, Any]]] = [
    (insp, "_TRIM_REMOVE_EXHAUSTED", insp._OUTPUT_SCHEMA["properties"]["results"]["items"]),
]


def _declared_trim_allowlists() -> set[tuple[str, str]]:
    """Every module-level ``_TRIM_*`` name assigned anywhere in the package,
    as ``(module, name)``. Read off the source with ``ast`` so nothing has to
    be imported for its constants to be seen."""
    package = Path(response_budget.__file__).resolve().parents[1]
    declared: set[tuple[str, str]] = set()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package.parent).with_suffix("")
        module = ".".join(relative.parts)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            declared.update(
                (module, target.id)
                for target in targets
                if isinstance(target, ast.Name) and target.id.startswith("_TRIM_")
            )
    return declared


class TestRungZeroAllowlists:
    """Rung 0 against the schemas it edits.

    It is the one rung that exempts content, so its keys are declared as data
    and checked here rather than reasoned about at the call site. A ``REMOVE``
    key must be optional — deleting a required key would make the emission fail
    the tool's own schema — and so must an ``EMPTY`` key, a block the rung
    empties of content, because the reply then leaves the empty block out.
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
            assert key not in required, f"{name} would leave out required key {key!r}"
            if "_EMPTY_" in name:
                kind = schema["properties"][key].get("type")
                assert kind in ("array", "object"), f"{name} empties {key!r}, which holds no block"

    def test_every_declared_allowlist_is_pinned(self):
        """The fail-closed half: a rung-0 list declared anywhere in the package
        and not listed above is an exemption nothing checks.

        Discovered from the source rather than from a list of modules: the
        receipt's allowlist lives in ``receipts``, which no tool registers in,
        so a hand-kept module list is exactly what would miss the next one.
        """
        assert _declared_trim_allowlists() == {
            (module.__name__, name) for module, name, _ in _TRIM_ALLOWLISTS
        }


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
                {"path": str(raw), "label": f"corner{index:02d}"} for index in range(_WIDE_SOURCES)
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
                    {"path": str(raw), "label": f"corner{index:02d}"}
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


# Three recipes over the same fan-out: three value lists, so one page limit caps
# three row surfaces at once.
_SURFACE_RECIPES: list[dict[str, Any]] = [
    {"key": f"v_{at}", "metric": "value", "expr": "V(out)", "at": at}
    for at in ("100u", "500u", "900u")
]


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
        rows = [{"a": "x" * 40} for _ in range(20)]
        rung = Rung(level=response_budget.RUNG_SHRINK, budget=500, measured=0)
        generous = response_budget.RowMeasure.of([rows], page=0)
        assert generous.fit_limit(20, rung) == 20
        starved = response_budget.RowMeasure.of([rows], page=100_000)
        assert starved.fit_limit(20, rung) == 1

    def test_a_limit_is_priced_on_every_surface_it_caps(self):
        surface = [{"a": "x" * 30} for _ in range(10)]
        rows = response_budget.estimate_tokens(surface)
        fixed = 100
        apart = response_budget.RowMeasure.of([surface] * 3, page=fixed + 3 * rows)
        pooled = response_budget.RowMeasure.of([surface * 3], page=fixed + 3 * rows)
        # Room for half of every surface's rows, and not one row more.
        room = 3 * rows // 2 + 1
        rung = Rung(
            level=response_budget.RUNG_SHRINK,
            budget=fixed + room + response_budget.NOTE_RESERVE_TOKENS,
            measured=0,
        )
        assert apart.affordable(rung) == 5
        # One pool of thirty rows affords fifteen, which as each surface's own
        # cap cuts none of them: the budget's rows, three times over.
        assert pooled.affordable(rung) > 10

    def test_largest_fitting_lands_on_a_limit_it_saw_fit(self):
        seen: list[int] = []

        def fits_up_to(bound: int):
            def fits(limit: int) -> bool:
                seen.append(limit)
                return limit <= bound

            return fits

        assert response_budget.largest_fitting(50, fits_up_to(17)) == 17
        assert 17 in seen
        assert response_budget.largest_fitting(50, fits_up_to(99)) == 50
        # Nothing fits: the floor, which is the one limit it may return unseen.
        assert response_budget.largest_fitting(50, fits_up_to(-1), floor=1) == 1

    def test_rung_zero_empties_only_the_blocks_it_names(self):
        container = {"named": [1, 2], "kept": [1]}
        response_budget.apply_trim(container, ("named",))
        assert container["kept"] == [1]
        assert container["named"] == []

    def test_rung_zero_reports_only_what_it_emptied_of_content(self):
        """A block that was already empty carried nothing, so it is not reported as cut."""
        container = {"named": [1, 2], "already": []}
        emptied = response_budget.apply_trim(container, ("named", "already"))
        assert emptied == ["named"]
        assert response_budget.apply_trim(container, ("named",)) == []

    def test_a_trim_that_emptied_nothing_is_not_a_degraded_response(self):
        """The note says presentation was reduced; with nothing removed it
        would be false, and its route would send the caller after nothing."""
        tidied = response_budget.Negotiated(
            data={"observations": []},
            rung=Rung(level=response_budget.RUNG_TRIM, budget=600, measured=900),
            estimate=900,
            max_rung=response_budget.RUNG_TRIM,
        )
        response_budget.attach_notes(tidied, response_budget.Notes(cut="cut", route="route"))
        assert tidied.data["observations"] == []

        cut = Rung(level=response_budget.RUNG_TRIM, budget=600, measured=900, cut=["echo"])
        emptied = response_budget.Negotiated(
            data={"observations": []},
            rung=cut,
            estimate=900,
            max_rung=response_budget.RUNG_TRIM,
        )
        notes = response_budget.Notes(cut="cut", route="raise budget", default_route="see rows")
        response_budget.attach_notes(emptied, notes)
        (note,) = emptied.data["observations"]
        assert "Emptied: echo." in note["detail"]
        # The caller set no budget, so the note sends them to none.
        assert note["detail"].endswith("see rows")
        assert "raise budget" not in note["detail"]

    def test_notes_extend_observations_rather_than_replacing_them(self):
        """A budget cuts presentation, so it may never overwrite a fact channel
        a tool already filled — the epilogue appends, on every tool."""
        rung = Rung(level=response_budget.RUNG_SHRINK, budget=600, measured=99_999)
        result = response_budget.Negotiated(
            data={"observations": [{"code": "prior"}], "hint": "keep me"},
            rung=rung,
            estimate=99_999,
        )
        notes = response_budget.Notes(cut="cut", route="route")
        response_budget.attach_notes(result, notes)
        codes = [o["code"] for o in result.data["observations"]]
        assert codes == ["prior", "budget_truncated", "budget_not_met"]
        # The note is structured content already; a hint copy would repeat it.
        assert result.data["hint"] == "keep me"

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
        plain = _request_digest(_wide_args(raw))
        budgeted = _request_digest(_wide_args(raw, budget=600))
        assert budgeted == plain

    async def test_the_row_view_is_part_of_the_request_a_cursor_reads(self, work_dir: Path):
        """A stored answer holds rows in one view, so a cursor reads them only
        for the request that gave it, include.fields included."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = _request_digest(
            _wide_args(raw, include={"per_run": {"limit": 5}, "fields": ["value"]})
        )
        second = _request_digest(
            _wide_args(raw, include={"per_run": {"limit": 5}, "fields": ["case_id"]})
        )
        assert first != second

    async def test_a_cursor_minted_under_a_budget_resumes_without_one(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The per_run cursor is checked against the request hash; a budget
        outside that hash is what lets these two calls be the same request."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        first = await _analysis(state_no_sim, raw, budget=900, include={"per_run": {"limit": 20}})
        cursor = first["results"]["loop"]["per_run"].get("next_cursor")
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
                "sources": [{"path": str(raw), "label": "dut"}],
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
        # A cut is said by cut, not by a ladder's note; the identity echo is a
        # list like any other, cut to what fits and never below one entry.
        assert _observation(data, "budget_truncated") is None
        assert data["source_hashes"]
        assert data["result_set_id"]

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
            assert "over its target of" in observation["detail"]
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

    async def test_a_budget_cuts_lists_and_keeps_every_opt_in(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A budget is a target the presenter cuts lists to; what the caller
        asked to include stays included, cut like any list."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        include = {"per_run": {"limit": 45}, "provenance": True, "signals_available": True}
        state_no_sim.config.default_budget = 0
        plain = await _analysis(state_no_sim, raw, include=include)
        assert plain["source_hashes"][0].get("raw_sha256") is not None

        data = await _analysis(
            state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS, include=include
        )
        assert _observation(data, "budget_truncated") is None
        assert data["source_hashes"][0].get("raw_sha256") is not None
        assert data["signals_available"]
        page = data["results"]["loop"]["per_run"]
        assert page["returned"] < plain["results"]["loop"]["per_run"]["returned"]
        assert page["returned"] == len(page["items"])
        assert page["next_cursor"] == data["cut"]["/results/loop/per_run/items"]["next"]

    async def test_shrunk_pages_still_reach_every_row(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The point of shrinking before assembly: a cursor from a shrunk page
        resumes at the row after the last one shown — no gap, no repeat."""
        state_no_sim.config.default_budget = 0
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
            cursor = page.get("next_cursor")
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
        state_no_sim.config.default_budget = 0
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        plain = await _spec_analysis(state_no_sim, raw)
        wide = plain["results"]["vout"]["spec"]
        assert wide["fail_count"] > 4, "fixture stopped producing failing cases"
        assert "truncated" not in wide["fail_cases"]

        data = await _spec_analysis(state_no_sim, raw, budget=response_budget.BUDGET_MIN_TOKENS)
        spec = data["results"]["vout"]["spec"]
        page = spec["fail_cases"]["items"]
        assert len(page) < wide["fail_cases"]["returned"]
        assert spec["fail_count"] == wide["fail_count"], "a count is a fact, not a page"
        assert spec["verdict"] == wide["verdict"]

    async def test_a_grouped_call_shrinks_its_groups_and_says_so(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A group list is cut like any list, and its cut names the cursor
        that reads the groups left out of the stored answer."""
        state_no_sim.config.default_budget = 0
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        plain = await _analysis(state_no_sim, raw, group_by=["r"])
        every_group = plain["results"]["loop"]["groups"]
        assert len(every_group) > 1

        data = await _analysis(
            state_no_sim, raw, group_by=["r"], budget=response_budget.BUDGET_MIN_TOKENS
        )
        entry = data["results"]["loop"]
        assert len(entry["groups"]) < len(every_group)
        cut = data["cut"]["/results/loop/groups"]
        assert (cut["total"], cut["returned"]) == (len(every_group), len(entry["groups"]))
        assert cut["next"]

    async def test_several_row_surfaces_share_the_budget_instead_of_each_taking_it(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The shrink rung measured the rows every surface showed as one pool,
        and the count it afforded became each surface's own cap, so three
        recipes' value lists each took the whole allowance: the response came
        back budget_not_met, telling the caller to narrow a request whose
        smaller page would have fitted."""
        state_no_sim.config.default_budget = 0
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        full = response_budget.estimate_tokens(
            await _analysis(state_no_sim, raw, recipes=_SURFACE_RECIPES)
        )

        # A sixth of the whole is below the floor now that the identity echo
        # keeps one entry where the trim rung emptied it.
        for divisor in (2, 3, 4):
            budget = full // divisor
            data = await _analysis(state_no_sim, raw, recipes=_SURFACE_RECIPES, budget=budget)
            assert _observation(data, "budget_not_met") is None, budget
            assert response_budget.estimate_tokens(data) <= budget
            assert all(entry.get("values") for entry in data["results"].values()), budget

    async def test_a_budget_below_the_floor_is_refused_at_the_edge(self):
        """Not a silent clamp: a number the ladder could never negotiate is a
        request error, so the caller learns the floor instead of guessing."""
        with pytest.raises(Exception, match="greater than or equal to 500"):
            AnalyzeResultsInput.model_validate(
                {"sources": [{"path": "x.raw"}], "recipes": [_WIDE_RECIPE], "budget": 1}
            )


@pytest.mark.asyncio
async def test_a_receipts_cursor_starts_after_the_rows_it_holds(
    state_no_sim: SessionState, work_dir: Path
):
    """The cut's cursor reads the job's record from the first row the reply
    left out: a status under the floor budget, then the runs it names."""
    job = make_experiment_job(state_no_sim, job_id="b_cursor", count=80, status="completed")
    data = await _jobs(
        state_no_sim,
        action="status",
        job_id=job.job_id,
        budget=response_budget.BUDGET_MIN_TOKENS,
    )
    page = data["runs"]
    cut = data["cut"]["/runs/items"]
    assert 1 <= page["returned"] == cut["returned"] < 80
    assert page["next_cursor"] == cut["next"]
    assert [row["case_id"] for row in page["items"]] == [
        case.case_id for case in job.cases[: page["returned"]]
    ]
    follow = await _jobs(
        state_no_sim,
        action="runs",
        job_id=job.job_id,
        cursor=cut["next"],
        budget=response_budget.BUDGET_MIN_TOKENS,
    )
    assert follow["cut"]["/items"]["offset"] == page["returned"]
    assert follow["items"][0]["case_id"] == job.cases[page["returned"]].case_id


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

    async def test_a_large_default_response_is_cut_with_a_cursor(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 500
        trimmed = await _analysis(state_no_sim, raw)
        assert trimmed["results"]["loop"]["values"]
        assert trimmed["cut"]["/results/loop/values"]["next"]
        assert _observation(trimmed, "budget_truncated") is None

    async def test_the_default_and_a_caller_budget_are_one_target(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The same number from the config or the call gives the same reply:
        there is no rung for the default to stop at."""
        raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
        state_no_sim.config.default_budget = 500
        defaulted = await _analysis(state_no_sim, raw, include={"per_run": {"limit": 20}})
        state_no_sim.config.default_budget = 0
        explicit = await _analysis(
            state_no_sim, raw, budget=500, include={"per_run": {"limit": 20}}
        )
        assert _stable(defaulted) == _stable(explicit)
        assert defaulted["results"]["loop"]["per_run"]["returned"] < 20

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
                "sources": [{"path": str(raw), "label": "dut"}],
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


_FIVE_TRACES = ["V(in)", "V(out)", "I(C1)", "I(R1)", "I(V1)"]


async def _waveform_reply(
    state: SessionState, raw: Path, *, budget: int | None = None, **recipe: Any
) -> dict[str, Any]:
    """Five traces of one transient, inline, as one ``waveform`` recipe."""
    extra = {} if budget is None else {"budget": budget}
    recipes = [{"key": "w", "metric": "waveform", "signals": _FIVE_TRACES, **recipe}]
    return await _analyze(state, raw, recipes, **extra)


def _inline_values(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [row["value"] for row in entry["values"]]


def _thinning_warnings(entry: dict[str, Any]) -> list[str]:
    return [warning for warning in entry.get("warnings", []) if "thinned" in warning]


@pytest.mark.asyncio
class TestWaveformSamplesFitTheBudget:
    """An inline waveform's samples were the one part of a reply no rung could
    reduce, so five traces of a 221-point transient came back several times
    over the budget. The samples are already a bounded min/max decimation, so
    the ladder thins them further to fit, keeping each bucket's extremes."""

    async def test_the_server_default_thins_an_unsized_waveform_to_fit(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        state_no_sim.config.default_budget = 0
        full = await _waveform_reply(state_no_sim, raw)
        state_no_sim.config.default_budget = 2000
        thinned = await _waveform_reply(state_no_sim, raw)

        assert response_budget.estimate_tokens(full) > 2000, "the premise: over the default"
        assert response_budget.estimate_tokens(thinned) <= 2000
        (whole,) = _inline_values(full["results"]["w"])
        (thin,) = _inline_values(thinned["results"]["w"])
        assert thin["points_total"] == whole["points_total"]
        assert thin["points_returned"] == len(thin["x"]) < whole["points_returned"]
        for before, after in zip(whole["series"], thin["series"], strict=True):
            assert after["signal"] == before["signal"]
            assert len(after["y"]) == len(thin["x"])
            assert min(after["y"]) == min(before["y"])
            assert max(after["y"]) == max(before["y"])
        (warning,) = _thinning_warnings(thinned["results"]["w"])
        assert "budget" in warning and "csv" in warning
        assert _observation(thinned, "budget_not_met") is None

    async def test_a_reply_too_long_thins_an_explicit_max_points_too(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """max_points is the most points the answer holds; a reply too long for
        its target shows fewer, and its cut gives the points held and shown."""
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        state_no_sim.config.default_budget = 2000
        data = await _waveform_reply(state_no_sim, raw, max_points=2000)

        assert response_budget.estimate_tokens(data) <= 2000
        (value,) = _inline_values(data["results"]["w"])
        assert value["points_returned"] < value["points_total"]
        cut = data["cut"]["/results/w/values/0/value/x"]
        assert (cut["total"], cut["returned"]) == (value["points_total"], len(value["x"]))
        assert "next" not in cut
        (warning,) = _thinning_warnings(data["results"]["w"])
        assert "csv" in warning
        # The budget the warning names sends every point the caller asked for.
        named = re.search(r"budget (\d+) sends every point", warning)
        assert named is not None, warning
        every = await _waveform_reply(
            state_no_sim, raw, budget=int(named.group(1)), max_points=2000
        )
        (whole,) = _inline_values(every["results"]["w"])
        assert whole["points_returned"] == whole["points_total"] == len(whole["x"])
        assert "cut" not in every

    async def test_a_caller_budget_thins_as_the_default_does(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        state_no_sim.config.default_budget = 0
        data = await _waveform_reply(state_no_sim, raw, budget=2000, max_points=2000)

        assert response_budget.estimate_tokens(data) <= 2000
        (value,) = _inline_values(data["results"]["w"])
        assert value["points_returned"] < value["points_total"]
        (warning,) = _thinning_warnings(data["results"]["w"])
        assert "budget" in warning
        assert _observation(data, "budget_truncated") is None

    async def test_receipts_thin_their_attached_waveform_to_fit(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        """A receipt's attached waveform is thinned under the server's default
        too, whatever max_points the request set, and every case's row is
        shown or reached by the cut's cursor."""
        budget = state_with_sim.config.default_budget
        recipe = {"key": "w", "metric": "waveform", "signals": ["V(in)", "V(out)"]}
        receipt = await _grid_receipt(state_with_sim, grid_deck, 4, None, recipes=[recipe])
        status = await _jobs(state_with_sim, action="status", job_id=receipt["job_id"])

        for data in (receipt, status):
            assert response_budget.estimate_tokens(data) <= budget
            entry = data["analysis"]["result"]["results"]["w"]
            values = _inline_values(entry)
            cut = data["analysis"]["result"].get("cut", {}).get("/results/w/values")
            assert len(values) == (4 if cut is None else cut["returned"])
            assert cut is None or (cut["total"] == 4 and cut["next"])
            for value in values:
                assert value["points_returned"] == len(value["x"]) < value["points_total"]
            (warning,) = _thinning_warnings(entry)
            assert "analyze_results with budget" in warning


@pytest.mark.asyncio
class TestJobsBudget:
    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(state_no_sim, 60)
        # The same target, once from the server's default and once from the
        # call: with no target at all an MCP reply holds 50 run rows.
        state_no_sim.config.default_budget = 1_000_000
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
            cursor = page.get("next_cursor")
            if cursor is None:
                break
        assert cursor is None
        assert seen == expected

    async def test_rows_keep_their_keys_at_every_rung(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A budget shrinks the page, never a row: a produced run's paths
        dropped under a budget read back as absent to a caller reading its
        log from the row."""
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
        assert page["returned"] < 60, "this budget is supposed to shrink the page"
        assert all(set(row) == set(plain["items"][0]) for row in rows)

    async def test_facts_and_handles_survive(self, state_no_sim: SessionState, work_dir: Path):
        job = _batch_with_runs(state_no_sim, 60)
        data = await _jobs(
            state_no_sim,
            action="runs",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert "failures" not in data
        assert data["job_id"] == job.job_id
        assert data["status"] == job.status
        # The cut says what the page holds; no note claims nothing was cut.
        cut = data["cut"]["/items"]
        assert cut["total"] == 60 and 1 <= cut["returned"] < 60
        assert _observation(data, "budget_truncated") is None


# ---------------------------------------------------------------------------
# The experiment receipt's floor
# ---------------------------------------------------------------------------

_GRID_DECK = "* grid\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"

# Three recipes with no reduction, so the attached analysis carries one row per
# run for each of them: a run page plus three row surfaces under one limit.
_GRID_RECIPES: list[dict[str, Any]] = [
    {"key": f"v_{at}", "metric": "value", "expr": "V(out)", "at": at}
    for at in ("100u", "500u", "900u")
]

# How far the floor may move between a 4-case and a 64-case grid: the digits
# of the counts it reports, not anything per case.
_FLOOR_DIGIT_SLACK = 16


# A signal no run carries: the recipe fails the same way on every run.
_ABSENT_SIGNAL_RECIPE: dict[str, Any] = {
    "key": "absent",
    "metric": "value",
    "expr": "V(nope)",
    "at": "900u",
}


@pytest.fixture
def grid_deck(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The grid's deck, run by a stand-in simulator that hands back a recorded raw+log."""
    recorded_fixture_simulator(monkeypatch)
    deck = work_dir / "grid.cir"
    deck.write_text(_GRID_DECK)
    return deck


async def _grid_receipt(
    state: SessionState,
    deck: Path,
    cases: int,
    budget: int | None,
    recipes: list[dict[str, Any]] = _GRID_RECIPES,
) -> dict[str, Any]:
    """A completed ``cases``-case grid over two-by-two by ``cases // 4`` values.

    ``budget=None`` sends none, so the server's default applies.
    """
    values = [f"{index}k" for index in range(1, cases // 4 + 1)]
    payload: dict[str, Any] = {
        "request_id": f"grid-{recipes[0]['key']}-{cases}-{budget}",
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
        "variations": [
            {"kind": "assign", "assign": {"R1": values, "C1": ["1u", "2u"], "V1": ["1", "2"]}}
        ],
        "analyze": {"recipes": recipes},
    }
    if budget is not None:
        payload["budget"] = budget
    _is_error, data = await submit_experiment(state, payload)
    assert data["status"] == "completed", data["hint"]
    assert data["completeness"]["produced"] == cases
    return data


def _assert_floor_flat(small: dict[str, Any], large: dict[str, Any]) -> None:
    """Two floors that differ by the digits of their counts, nothing per case.
    ``runs_not_shown`` is bounded on its own (``NOT_SHOWN_BOUND_TOKENS``), so
    it is measured apart."""
    for floor in (small, large):
        note = floor.get("runs_not_shown")
        if note is not None:
            assert (
                response_budget.estimate_tokens({"runs_not_shown": note})
                <= receipts_mod.NOT_SHOWN_BOUND_TOKENS
            )
    lean = [{k: v for k, v in floor.items() if k != "runs_not_shown"} for floor in (small, large)]
    sizes = (response_budget.estimate_tokens(lean[0]), response_budget.estimate_tokens(lean[1]))
    assert sizes[1] - sizes[0] <= _FLOOR_DIGIT_SLACK, sizes


@pytest.mark.asyncio
class TestReceiptFloor:
    """The most degraded receipt costs the same whatever the job's size.

    A budget is set because a receipt is large, and a receipt is large because
    the job ran many cases. A floor that carried a row per case — a run row, an
    attributed value per recipe, an identity echo per run — grew with exactly
    the number that made the caller reach for a budget, so a big grid came back
    over any budget it could set. At the floor one run row stays, the rest are
    counted in ``cut``, and its cursor reads every one from the job's record.
    """

    async def test_the_floor_does_not_grow_with_the_case_count(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        floor = response_budget.BUDGET_MIN_TOKENS

        receipts = {
            cases: await _grid_receipt(state_with_sim, grid_deck, cases, floor)
            for cases in (4, 64)
        }
        statuses = {
            cases: await _jobs(
                state_with_sim, action="status", job_id=data["job_id"], budget=floor
            )
            for cases, data in receipts.items()
        }

        _assert_floor_flat(receipts[4], receipts[64])
        _assert_floor_flat(statuses[4], statuses[64])
        large = receipts[64]
        # Counted, not carried: completeness and the cut say how many runs
        # there are; one row stays, and no per-run echo is inline.
        assert large["completeness"]["expanded"] == large["runs"]["total"] == 64
        assert large["runs"]["returned"] == large["cut"]["/runs/items"]["returned"] >= 1
        assert len(large["analysis"]["result"]["source_hashes"]) == 1
        assert large["analysis"]["result"]["coverage"]["runs_analyzed"] == 64
        assert _observation(large, "budget_truncated") is None
        assert 'jobs(action="runs"' in large["hint"]

        # The cursor is the route back to every row, in order.
        seen = [row["case_id"] for row in large["runs"]["items"]]
        cursor = large["runs"].get("next_cursor")
        while cursor is not None:
            page = await _jobs(
                state_with_sim, action="runs", job_id=large["job_id"], cursor=cursor
            )
            seen.extend(row["case_id"] for row in page["items"])
            cursor = page.get("next_cursor")
        assert seen == [f"dut-case-{index:04d}" for index in range(64)]

    async def test_a_recipe_failing_on_every_run_is_one_counted_row(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        """Failures are a fact channel no rung trims, so a failure per run kept
        the floor growing with the job: a signal no run carries is one message
        per run, identical but for ``where``. It is one reason, so one row,
        counted — on the receipt and on analyze_results over the job alike."""
        floor = response_budget.BUDGET_MIN_TOKENS
        # The row names its places up to a cap, so the floor stops moving once
        # the job has more runs than that: both sizes here are past it.
        sizes = (16, 64)
        assert sizes[0] > analyze_mod._FAILURE_WHERE_CAP
        # The places, in run order, capped: the count says how many more.
        places = [
            f"experiment:dut-case-{index:04d}" for index in range(analyze_mod._FAILURE_WHERE_CAP)
        ]

        receipts = {
            cases: await _grid_receipt(
                state_with_sim, grid_deck, cases, floor, recipes=[_ABSENT_SIGNAL_RECIPE]
            )
            for cases in sizes
        }

        _assert_floor_flat(receipts[sizes[0]], receipts[sizes[1]])
        for cases, data in receipts.items():
            (row,) = data["analysis"]["result"]["failures"]
            assert row["code"] == "recipe_failed"
            assert row["count"] == cases
            assert row["wheres"] == places
            assert row["where"] == places[0]

        standalone = await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {
                    "sources": [{"job_id": receipts[64]["job_id"]}],
                    "recipes": [_ABSENT_SIGNAL_RECIPE],
                }
            ),
            state_with_sim,
        )
        data = standalone.structured_content
        assert data is not None
        jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)
        (row,) = data["failures"]
        assert row["count"] == 64
        assert data["coverage"]["runs_requested"] == 64
        assert data["coverage"]["runs_analyzed"] == 0
        assert data["outcome"] != "complete"

    async def test_a_budget_the_rows_can_share_keeps_as_many_as_fit(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        """The shrink rung's estimate counts every surface's rows but caps each
        surface at the one limit it returns, so for a receipt with several row
        surfaces it priced a page that cut nothing as one that fit. The rung
        settles on a measured page instead."""
        budget = 6000

        data = await _grid_receipt(state_with_sim, grid_deck, 24, budget)

        assert _observation(data, "budget_not_met") is None
        assert response_budget.estimate_tokens(data) <= budget
        assert 0 < data["runs"]["returned"] < 24

    async def test_a_jobs_receipt_keeps_as_many_rows_as_the_run_receipt(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        """One receipt, one measure. jobs(status) left the attached analysis's
        rows out of its estimate, charging them as fixed cost, so it started its
        search at one row and returned that where run_experiments, for the same
        job under the same budget, returned as many as fit."""
        budget = 6000
        receipt = await _grid_receipt(state_with_sim, grid_deck, 24, budget)

        status = await _jobs(
            state_with_sim, action="status", job_id=receipt["job_id"], budget=budget
        )

        assert _observation(status, "budget_not_met") is None
        assert response_budget.estimate_tokens(status) <= budget
        # The envelopes differ by a few keys, which may cost a row either way.
        assert abs(status["runs"]["returned"] - receipt["runs"]["returned"]) <= 1, (
            status["runs"]["returned"],
            receipt["runs"]["returned"],
        )

    async def test_the_server_default_cuts_the_attached_analysis_beside_the_rows(
        self, state_with_sim: SessionState, grid_deck: Path
    ):
        """The attached analysis is cut under the receipt's target like its
        run rows: each of its lists at least one entry, the rest behind the
        cursor its cut names."""
        state_with_sim.config.default_budget = response_budget.BUDGET_MIN_TOKENS

        data = await _grid_receipt(state_with_sim, grid_deck, 4, None, recipes=_GRID_RECIPES[:1])

        attached = data["analysis"]["result"]
        assert attached["source_hashes"]
        assert attached["results"]["v_100u"]["values"]
        assert any("next" in entry for entry in attached["cut"].values())
        assert 1 <= data["runs"]["returned"] == data["cut"]["/runs/items"]["returned"] <= 4
        assert _observation(data, "budget_truncated") is None


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
    wire_disabled = asyncio.run(_inspect(state_no_sim, [copy.deepcopy(query)]))
    state_no_sim.config.default_budget = 500
    wire = asyncio.run(_inspect(state_no_sim, [copy.deepcopy(query)]))
    assert _stable(wire) != _stable(wire_disabled), "the default did not engage on the wire"


def test_the_server_default_sends_no_caller_to_a_budget(
    state_no_sim: SessionState, work_dir: Path
):
    """A caller who set no budget has none to raise, and the API refuses the
    field outright. Checked on jobs(list), the collected surface that keeps the
    observations its pages carried: its trim only drops an empty 'analysis'
    block, which takes nothing a caller reads, so neither interface is told
    anything was reduced."""
    for index in range(6):
        job = _batch_with_runs(state_no_sim, 20)
        job.job_id = f"b_budget_{index}"
        state_no_sim.all_jobs[job.job_id] = job
    state_no_sim.config.default_budget = 100

    complete = SyncApi(state_no_sim).jobs(action="list")
    wire = asyncio.run(_jobs(state_no_sim, action="list"))

    for data in (wire, complete):
        assert _observation(data, "budget_truncated") is None
        assert not [o for o in data.get("observations", []) if "budget'" in o.get("detail", "")]


@pytest.mark.asyncio
class TestInspectBudget:
    async def test_a_met_budget_changes_nothing(self, state_no_sim: SessionState, work_dir: Path):
        path = _many_component_netlist(work_dir, 90)
        query = {"kind": "components", "path": str(path)}
        plain = await _inspect(state_no_sim, [query])
        met = await _inspect(state_no_sim, [query], budget=1_000_000)
        assert _stable(met) == _stable(plain)

    async def test_a_batch_shares_the_budget_instead_of_each_query_taking_it(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Each query's page is its own surface under the one shrunk limit, so
        a batch priced as one pool of rows gave every query the whole allowance
        and came back over a budget a smaller page per query would have met."""
        state_no_sim.config.default_budget = 0
        path = _many_component_netlist(work_dir, 90)
        batch = [{"kind": "components", "path": str(path)} for _ in range(3)]
        full = response_budget.estimate_tokens(await _inspect(state_no_sim, batch))

        for divisor in (2, 3, 4):
            budget = full // divisor
            data = await _inspect(state_no_sim, batch, budget=budget)
            assert _observation(data, "budget_not_met") is None, budget
            assert response_budget.estimate_tokens(data) <= budget
            assert all(item["data"]["components"] for item in data["results"]), budget

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
        # The note is on observations; the hint stays the batch's own.
        assert data["hint"] == (
            "1 of 2 queries failed; see each result's 'error.code'. Other queries "
            "returned normally."
        )

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

    async def test_run_receipt_rows_stay_objects(self, state_no_sim: SessionState):
        job = make_experiment_job(state_no_sim, job_id="b_objects", count=80)
        state_no_sim.config.default_budget = 0
        plain = await _jobs(state_no_sim, action="status", job_id=job.job_id)
        data = await _jobs(
            state_no_sim,
            action="status",
            job_id=job.job_id,
            budget=response_budget.BUDGET_MIN_TOKENS,
        )
        assert data["runs"]["returned"] < 80, "this budget is supposed to cut the rows"
        assert _columns_siblings(data) == []
        _assert_object_rows(
            data["runs"]["items"], set(plain["runs"]["items"][0]), where="receipt runs"
        )

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
    # And the hint names the lever that moved. The caller asked for the cap, so
    # telling it to raise 'limit' would send it back to a knob it already
    # maxed out over a page the budget cut.
    hint = tight["results"][0]["data"]["hint"]
    assert "budget" in hint and "Raise 'limit'" not in hint
