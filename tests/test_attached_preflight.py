"""What the attached analyze block accepts at submission, it can run after the runs.

The block is an ``analyze_results`` request run once the job's runs finish, so
a fault in it found only then costs the whole simulation, and the corrected
block is a different request whose retry runs every case again. So a recipe
``analyze_results`` would refuse is refused at submission as that tool reports
it (``recipe_invalid``, stage ``validate``, not retryable), and a field that
could only fail after the runs, ``include.per_run.cursor``, is refused there
too; every other field the block accepts runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests.conftest import (
    LIVENESS_S,
    call_tool_params,
    fake_request_context,
    fake_simulator,
    make_experiment_job,
    stage_recorded_fixture,
)

_INVALID_RECIPES = [
    pytest.param({"key": "k", "metric": "signal_stats"}, id="missing-field"),
    pytest.param({"key": "k", "metric": "no_such_metric"}, id="unknown-metric"),
    pytest.param(
        {"key": "k", "metric": "signal_stats", "signal": "V(out)", "not_a_field": 1},
        id="unknown-field",
    ),
]


def _deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text("V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n", encoding="utf-8")
    return deck


async def _submit(state: SessionState, deck: Path, analyze_block: dict[str, Any]) -> Any:
    args = experiments.RunExperimentsInput.model_validate(
        {
            "request_id": "attached-preflight",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": LIVENESS_S},
            "analyze": analyze_block,
        }
    )
    return await experiments.handle_run_experiments(args, state)


@pytest.mark.parametrize("recipe", _INVALID_RECIPES)
async def test_an_invalid_recipe_is_reported_as_analyze_results_reports_it(
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe: dict[str, Any],
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    page = (
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {"sources": [{"path": str(raw), "label": "rc"}], "recipes": [recipe]}
            ),
            state_with_sim,
        )
    ).structured_content
    assert page is not None
    (failure,) = page["failures"]
    assert (failure["code"], failure["stage"]) == ("recipe_invalid", "validate")

    submissions: list[str] = []
    fake_simulator(monkeypatch, submissions)
    result = await _submit(state_with_sim, _deck(work_dir), {"recipes": [recipe]})
    assert result.is_error
    assert result.structured_content is not None
    error = result.structured_content["error"]
    assert (error["code"], error["stage"]) == (failure["code"], failure["stage"])
    assert error["retryable"] is False
    assert error["commit_state"] == "not_started"
    assert submissions == []


async def test_a_per_run_cursor_is_refused_at_submission(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = {
        "recipes": [{"key": "k", "metric": "signal_stats", "signal": "V(out)"}],
        "include": {"per_run": {"cursor": "abc"}},
    }
    with pytest.raises(ValidationError, match="no page to resume at submission"):
        experiments.AttachedAnalysis.model_validate(block)

    submissions: list[str] = []
    fake_simulator(monkeypatch, submissions)
    result = await call_tool(
        fake_request_context(state_with_sim),
        call_tool_params(
            "run_experiments",
            {"circuits": [{"path": str(_deck(work_dir))}], "analyze": block},
        ),
    )
    assert result.is_error
    assert result.structured_content is not None
    assert result.structured_content["code"] == "invalid_arguments"
    assert "analyze.include.per_run.cursor" in result.structured_content["error"]
    assert submissions == []


async def test_every_field_the_block_accepts_runs_after_the_runs(
    state_with_sim: SessionState, work_dir: Path
) -> None:
    """The block as accepted at submission, stored as the job keeps it and run
    over the job's runs as the analysis stage runs it, answers."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    block = experiments.AttachedAnalysis.model_validate(
        {
            "recipes": [
                {"key": "k", "metric": "signal_stats", "signal": "V(out)"},
                {"key": "m", "metric": "summary"},
            ],
            "group_by": ["circuit"],
            "all_steps": True,
            "include": {
                "per_run": {"limit": 10},
                "outliers": True,
                "signals_available": True,
                "fields": ["rms", "case_id"],
            },
        }
    )
    experiments._validate_attached_analysis(block)  # pyright: ignore[reportPrivateUsage]
    stored_request = experiments.RunExperimentsInput.model_validate(
        {"circuits": [{"path": "rc.cir"}], "analyze": block.model_dump(mode="json")}
    ).strip_presentation()["analyze"]
    job = make_experiment_job(state_with_sim, job_id="exp_preflight_0001", raw=raw)
    payload = experiments._attached_analysis_payload(  # pyright: ignore[reportPrivateUsage]
        job.job_id, stored_request
    )
    stored = await analyze.capture_attached_analysis(
        AnalyzeResultsInput.model_validate(payload), state_with_sim
    )
    shown = analyze.render_attached_analysis(stored, fields=block.include and block.include.fields)
    assert shown["failures"] == []
    assert set(shown["results"]) == {"k", "m"}
    rows = shown["results"]["k"]["per_run"]["items"]
    assert rows and all(set(row) == {"case_id", "value"} and "rms" in row["value"] for row in rows)
