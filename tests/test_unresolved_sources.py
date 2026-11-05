"""A source that names nothing is reported one way by every surface.

An unknown ``job_id`` is ``job_not_found`` and a path with no file is
``result_file_not_found`` (``docs/design/mcp_surface.md``, "Error codes"),
each with its kind's one recovery step (``errors.ERROR_HINTS``), whichever
tool or Python API method was handed it. ``analyze_results`` once called both
``source_unavailable``, in two different channels, with no hint; a path with
no file was then given the recovery for a job's results, which sent the
caller to check a run it never made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import ApiCallError
from ltspice_mcp.errors import (
    ERROR_HINTS,
    JobNotFoundError,
    ResultError,
    ResultFileNotFoundError,
    error_hint,
)
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, call_tool_params, fake_request_context

_UNKNOWN_JOB = "exp_nope_123"
#: What each kind of source that names nothing is, and the step it gets.
_KINDS: dict[str, tuple[str, str]] = {
    "unknown_job": ("job_not_found", ERROR_HINTS[JobNotFoundError]),
    "missing_file": ("result_file_not_found", ERROR_HINTS[ResultFileNotFoundError]),
    "missing_log": ("result_file_not_found", ERROR_HINTS[ResultFileNotFoundError]),
}


def _source(kind: str, work_dir: Path) -> dict[str, str]:
    if kind == "unknown_job":
        return {"job_id": _UNKNOWN_JOB}
    return {"path": str(work_dir / ("nope.log" if kind == "missing_log" else "nope.raw"))}


async def _call(state: SessionState, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await call_tool(fake_request_context(state), call_tool_params(tool, arguments))
    assert result.structured_content is not None
    return result.structured_content


def _reported(code: str, hint: str | None) -> tuple[str, str | None]:
    return code, hint


async def _mcp_answers(kind: str, state: SessionState, work_dir: Path) -> dict[str, tuple]:
    """Each MCP surface's code and hint for one source of ``kind``."""
    source = _source(kind, work_dir)
    answers: dict[str, tuple] = {}

    analyzed = await _call(
        state,
        "analyze_results",
        {
            "sources": [{**source, "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "1m"}],
        },
    )
    [row] = analyzed["coverage"]["missing_cases"]["items"]
    answers["analyze_results"] = _reported(row["code"], row.get("hint"))

    plotted = await _call(state, "plot_waveform", {**source, "signals": ["V(out)"]})
    answers["plot_waveform"] = _reported(plotted["code"], plotted.get("hint"))

    # The signals view reads a raw, the measurements view a log.
    for view in ("signals", "measurements"):
        inspected = await _call(
            state, "inspect", {"queries": [{"kind": "results", "view": view, **source}]}
        )
        [item] = inspected["results"]
        answers[f"inspect_{view}"] = _reported(item["error"]["code"], item["error"].get("hint"))

    if kind == "unknown_job":
        for action in ("status", "wait", "runs", "cancel", "place_beside_sheet"):
            jobs = await _call(state, "jobs", {"action": action, "job_id": _UNKNOWN_JOB})
            # A jobs hint is its message, then the step.
            message = jobs["error"]["message"]
            assert jobs["hint"].startswith(message), jobs
            step = jobs["hint"][len(message) :].strip() or None
            answers[f"jobs_{action}"] = _reported(jobs["error"]["code"], step)
    return answers


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(_KINDS))
async def test_every_tool_reports_a_source_that_names_nothing_one_way(
    kind: str, state_no_sim: SessionState, work_dir: Path
):
    code, step = _KINDS[kind]
    answers = await _mcp_answers(kind, state_no_sim, work_dir)
    # The first reply of a session carries the guide reminder after its hint.
    answers = {
        surface: (got, hint[: len(step)] if hint else hint)
        for surface, (got, hint) in answers.items()
    }
    assert answers == dict.fromkeys(answers, (code, step)), answers


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_every_api_method_reports_a_source_that_names_nothing_one_way(
    kind: str, state_no_sim: SessionState, work_dir: Path
):
    code, step = _KINDS[kind]
    api = SyncApi(state_no_sim)
    source = _source(kind, work_dir)
    answers: dict[str, tuple] = {}

    analyzed = api.analyze_results(
        sources=[{**source, "label": "dut"}],
        recipes=[{"key": "v", "metric": "value", "expr": "V(out)", "at": "1m"}],
    )
    [row] = analyzed["coverage"]["missing_cases"]["items"]
    answers["analyze_results"] = (row["code"], row.get("hint"))

    target: dict[str, Any] = (
        {"job_id": _UNKNOWN_JOB} if kind == "unknown_job" else {"path": source["path"]}
    )
    for method in (api.load_raw, api.measurements, api.device_op_points):
        with pytest.raises(ResultError) as raised:
            method(**target)
        answers[method.__name__] = (raised.value.code, error_hint(raised.value))

    if kind == "unknown_job":
        with pytest.raises(ApiCallError) as failed:
            api.jobs(action="status", job_id=_UNKNOWN_JOB)
        error = failed.value.payload["error"]
        hint = failed.value.payload["hint"]
        answers["jobs"] = (error["code"], hint[len(error["message"]) :].strip())

    assert answers == dict.fromkeys(answers, (code, step)), answers


@pytest.mark.asyncio
async def test_a_known_and_an_unknown_job_in_one_call_are_told_apart(
    state_no_sim: SessionState, work_dir: Path
):
    """The coverage row names the source that resolved to nothing; the rest of
    the call goes on."""
    analyzed = await _call(
        state_no_sim,
        "analyze_results",
        {
            "sources": [
                {"job_id": _UNKNOWN_JOB, "label": "typo"},
                {"path": str(work_dir / "nope.raw"), "label": "gone"},
            ],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "1m"}],
        },
    )
    rows = {row["label"]: row for row in analyzed["coverage"]["missing_cases"]["items"]}
    assert rows["typo"]["code"] == "job_not_found" and _UNKNOWN_JOB in rows["typo"]["detail"]
    assert rows["gone"]["code"] == "result_file_not_found" and "nope.raw" in rows["gone"]["detail"]
    assert analyzed.get("failures", []) == []
