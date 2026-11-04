"""A value above a cap is served at the cap and said so, never refused.

A cap bounds what one call costs, so asking for more of the same thing gets
the most the call will serve and a note naming the cap. The hygiene check
holds every tool-input page size, point count and wait to that rule; the rest
send a value over each cap through its tool and read the note.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import annotated_types
import pytest
from pydantic import BaseModel

from ltspice_mcp.lib.model_fields import models_in
from ltspice_mcp.lib.recipes import WAVEFORM_MAX_POINTS_CAP
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analysis, inspect_tools, run_code
from ltspice_mcp.tools._base import registry
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests.conftest import FIXTURES_DIR

#: The argument names whose cap bounds a call's cost: a page size, a point
#: count, a wait.
_CAPPED_NAMES = frozenset({"limit", "max_points", "timeout_s", "wait_s"})

#: Fields of those names that may refuse a value above their bound, with why.
_REFUSING_ALLOWED: dict[tuple[str, str], str] = {}


def _every_input_model() -> set[type[BaseModel]]:
    import ltspice_mcp.tools  # noqa: F401  (registers every tool)

    pending: list[type[BaseModel]] = [
        r.input_model
        for r in registry._registered  # pyright: ignore[reportPrivateUsage]
        if r.input_model is not None
    ]
    seen: set[type[BaseModel]] = set()
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending += list(model.__dict__.get("VARIANTS", ()))
        for field in model.model_fields.values():
            pending += models_in(field.annotation)
    return seen


def test_no_capped_argument_refuses_a_value_above_its_cap() -> None:
    models = _every_input_model()
    found = [
        (model, name) for model in models for name in model.model_fields if name in _CAPPED_NAMES
    ]
    assert len(found) >= 8, "the walk found too few capped arguments to be checking anything"
    refusing = [
        f"{model.__name__}.{name}"
        for model, name in found
        if (model.__name__, name) not in _REFUSING_ALLOWED
        and any(
            isinstance(meta, (annotated_types.Le, annotated_types.Lt))
            for meta in model.model_fields[name].metadata
        )
    ]
    assert not refusing, (
        "these refuse a value above their bound instead of holding it with a note "
        "(NotedModel.hold_to_cap): " + ", ".join(sorted(refusing))
    )


def _note(name: str, asked: int | float, cap: int | float, unit: str = "") -> str:
    def spelled(value: int | float) -> str:
        return f"{value:d}" if isinstance(value, int) else f"{value:g}"

    return (
        f"{name}={spelled(asked)}{unit} is above its cap of {spelled(cap)}{unit}; "
        f"{spelled(cap)}{unit} was used."
    )


def _raw(work_dir: Path) -> Path:
    raw = work_dir / "rc.raw"
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.log", work_dir / "rc.log")
    return raw


async def _inspect(state: SessionState, query: dict[str, Any]) -> dict[str, Any]:
    args = inspect_tools.InspectInput.model_validate({"queries": [query]})
    data = (await inspect_tools.handle_inspect(args, state)).structured_content
    assert data is not None
    (item,) = data["results"]
    return item


class TestInspect:
    async def test_a_results_limit_above_its_cap_is_held(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        item = await _inspect(
            state_no_sim, {"kind": "results", "view": "signals", "path": str(raw), "limit": 500}
        )
        assert item["ok"], item
        assert item["data"]["warnings"] == [_note("limit", 500, inspect_tools.RESULTS_LIMIT_CAP)]

    async def test_a_reference_limit_above_its_cap_is_held(
        self, state_no_sim: SessionState
    ) -> None:
        item = await _inspect(state_no_sim, {"kind": "reference", "query": "thd", "limit": 500})
        assert item["ok"], item
        assert item["data"]["warnings"] == [_note("limit", 500, inspect_tools.REFERENCE_LIMIT_CAP)]

    async def test_a_limit_within_its_cap_says_nothing(self, state_no_sim: SessionState) -> None:
        item = await _inspect(state_no_sim, {"kind": "reference", "query": "thd", "limit": 3})
        assert item["ok"] and "warnings" not in item["data"]


class _RecordingWorker:
    """Stands in for the snippet worker process: records the bound it is given."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    async def run(self, code: str, timeout_s: float, reset: bool) -> dict[str, Any]:
        self.timeouts.append(timeout_s)
        return {
            "status": "ok",
            "exec_seq": 1,
            "worker_pid": None,
            "stdout": "",
            "stderr": "",
            "result": None,
            "truncated": False,
            "chars_dropped": 0,
            "elapsed_s": 0.0,
            "error": None,
            "running": None,
            "worker_restarted": None,
            "hint": None,
        }


class TestRunCode:
    async def test_a_timeout_above_its_cap_is_held(self, state_no_sim: SessionState) -> None:
        worker = _RecordingWorker()
        state_no_sim.code_worker = worker  # type: ignore[assignment]
        args = run_code.RunCodeInput.model_validate({"code": "1", "timeout_s": 1000})
        result = await run_code.handle_run_code(args, state_no_sim)
        data = result.structured_content
        assert data is not None
        assert worker.timeouts == [run_code.RUN_CODE_TIMEOUT_CAP_S]
        note = _note("timeout_s", 1000.0, run_code.RUN_CODE_TIMEOUT_CAP_S, "s")
        assert data["warnings"] == [note]
        assert data["hint"].startswith(note)


class TestPlotWaveform:
    async def test_max_points_above_its_cap_is_held_and_said(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        asked = WAVEFORM_MAX_POINTS_CAP + 1_000_000
        args = analysis.PlotWaveformInput.model_validate(
            {"path": str(raw), "signals": ["V(out)"], "max_points": asked, "open": False}
        )
        data = (await analysis.handle_plot_waveform(args, state_no_sim)).structured_content
        assert data is not None
        assert data["max_points"] == WAVEFORM_MAX_POINTS_CAP
        assert data["warnings"] == [_note("max_points", asked, WAVEFORM_MAX_POINTS_CAP)]


class TestWaveformRecipe:
    @pytest.mark.parametrize("asked", [50_000, WAVEFORM_MAX_POINTS_CAP + 1])
    async def test_max_points_above_the_served_cap_is_held_and_said(
        self, state_no_sim: SessionState, work_dir: Path, asked: int
    ) -> None:
        raw = _raw(work_dir)
        cap = min(WAVEFORM_MAX_POINTS_CAP, state_no_sim.config.max_points_returned)
        args = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "rc"}],
                "recipes": [
                    {"key": "w", "metric": "waveform", "signals": ["V(out)"], "max_points": asked}
                ],
            }
        )
        data = (await handle_analyze_results(args, state_no_sim)).structured_content
        assert data is not None
        entry = data["results"]["w"]
        assert entry["warnings"] == [_note("max_points", asked, cap)]
