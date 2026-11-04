"""A malformed cursor is the same failure on every tool that takes one.

Each argument that resumes a page is sent a cursor no tool minted, through the
server's own dispatch. Every one answers ``invalid_cursor``, and a call-level
refusal's hint sends the caller back to that tool's own cursor rather than to
a job check or a netlist lint.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import InvalidCursorError
from ltspice_mcp.lib import result_store
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FIXTURES_DIR,
    call_tool_params,
    fake_request_context,
    make_experiment_job,
)

GARBAGE = "garbage"


async def _call(state: SessionState, tool: str, arguments: dict[str, Any]) -> Any:
    return await call_tool(fake_request_context(state), call_tool_params(tool, arguments))


def _assert_invalid_cursor(result: Any) -> dict[str, Any]:
    assert result.is_error
    data = result.structured_content
    assert data is not None
    error = data.get("error")
    # jobs and edit_schematic report inside their envelope; the others raise.
    code = error["code"] if isinstance(error, dict) else data["code"]
    assert code == "invalid_cursor", data
    return data


def _raw(work_dir: Path) -> Path:
    raw = work_dir / "rc.raw"
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.log", work_dir / "rc.log")
    return raw


_RECIPE = {"key": "k", "metric": "signal_stats", "signal": "V(out)"}


class TestCallLevelCursors:
    async def test_analyze_results_per_run_cursor(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        data = _assert_invalid_cursor(
            await _call(
                state_no_sim,
                "analyze_results",
                {
                    "sources": [{"path": str(raw), "label": "rc"}],
                    "recipes": [_RECIPE],
                    "include": {"per_run": {"cursor": GARBAGE}},
                },
            )
        )
        assert "next_cursor" in data["hint"]
        assert "jobs" not in data["hint"]

    async def test_analyze_results_continue_cursor(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        first = await _call(
            state_no_sim,
            "analyze_results",
            {
                "sources": [{"path": str(raw), "label": "rc"}],
                "recipes": [_RECIPE],
                "include": {"per_run": {"limit": 1}},
            },
        )
        assert not first.is_error
        result_set_id = first.structured_content["result_set_id"]
        data = _assert_invalid_cursor(
            await _call(
                state_no_sim,
                "analyze_results",
                {"continue": {"result_set_id": result_set_id, "cursor": GARBAGE}},
            )
        )
        assert "next_cursor" in data["hint"]

    async def test_jobs_list_cursor(self, state_no_sim: SessionState) -> None:
        data = _assert_invalid_cursor(
            await _call(state_no_sim, "jobs", {"action": "list", "cursor": GARBAGE})
        )
        assert data["error"]["stage"] == "pagination"
        assert data["error"].get("retryable", False) is False
        assert "next_cursor" in data["hint"]

    async def test_jobs_runs_cursor(self, state_no_sim: SessionState) -> None:
        job = make_experiment_job(state_no_sim, job_id="exp_cursor_0001", count=2)
        data = _assert_invalid_cursor(
            await _call(
                state_no_sim, "jobs", {"action": "runs", "job_id": job.job_id, "cursor": GARBAGE}
            )
        )
        assert data["error"]["stage"] == "pagination"
        assert "next_cursor" in data["hint"]

    @pytest.mark.parametrize("view", ["pin_legend", "label_only_pins", "touched", "preexisting"])
    async def test_edit_schematic_view_cursor(
        self, state_no_sim: SessionState, work_dir: Path, view: str
    ) -> None:
        data = _assert_invalid_cursor(
            await _call(
                state_no_sim,
                "edit_schematic",
                {
                    "path": str(work_dir / "sheet.asc"),
                    "ops": [],
                    "return_views": ["pin_legend"],
                    "view_cursors": {view: GARBAGE},
                },
            )
        )
        assert "next_cursor" in data["hint"]
        assert "verify_circuit" not in data["hint"]


class TestItemLevelCursors:
    async def test_inspect_results_cursor(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        result = await _call(
            state_no_sim,
            "inspect",
            {
                "queries": [
                    {"kind": "results", "view": "signals", "path": str(raw), "cursor": GARBAGE}
                ]
            },
        )
        data = result.structured_content
        assert data is not None
        (item,) = data["results"]
        assert item["ok"] is False
        assert item["error"]["code"] == "invalid_cursor"


class TestDecoders:
    """Every decoder of an analysis cursor raises the one type."""

    @pytest.mark.parametrize(
        "decode",
        [
            result_store.cursor_result_set_id,
            result_store.cursor_view,
            lambda cursor: result_store.reencode_cursor(cursor, view_fields=None),
        ],
    )
    def test_a_garbage_analysis_cursor_raises_the_cursor_error(self, decode: Any) -> None:
        with pytest.raises(InvalidCursorError):
            decode(GARBAGE)
