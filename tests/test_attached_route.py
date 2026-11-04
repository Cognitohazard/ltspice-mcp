"""A failure that ends in an ``api`` snippet says where it runs on every MCP page.

``analyze_results`` follows such a message with where the snippet runs for the
session's tools. The attached analysis of ``run_experiments`` is the same
engine, stored when the job's runs finish and shown later, by whichever
session asks; it says the same thing when it is shown, and the Python API,
which reads the message as is, gets it without.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ltspice_mcp.errors import python_route_text
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import (
    AnalyzeResultsInput,
    capture_attached_analysis,
    handle_analyze_results,
    render_attached_analysis,
)
from tests.conftest import call_tool_params, fake_request_context, stage_recorded_fixture

_EXPRESSION = "V(out)*I(R1)"


def _request(raw: Path) -> dict[str, Any]:
    return {
        "sources": [{"path": str(raw), "label": "rc"}],
        "recipes": [{"key": "expr", "metric": "signal_stats", "signal": _EXPRESSION}],
    }


async def test_the_attached_analysis_says_where_the_snippet_runs(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    served = frozenset(state_no_sim.tool_dispatch)
    route = python_route_text(served)

    page = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(_request(raw)), state_no_sim
    )
    assert page.structured_content is not None
    (failure,) = page.structured_content["failures"]
    assert failure["message"].endswith(route)

    stored = await capture_attached_analysis(
        AnalyzeResultsInput.model_validate(_request(raw)), state_no_sim
    )
    shown = render_attached_analysis(stored, fields=None, served=served)
    assert shown["failures"] == [failure]

    # The Python API reads the message as is, as api.analyze_results does.
    plain = render_attached_analysis(stored, fields=None)
    (bare,) = plain["failures"]
    assert f"{bare['message']} {route}" == failure["message"]
    assert "python_route" not in bare


async def test_the_route_follows_the_session_that_shows_it(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    stored = await capture_attached_analysis(
        AnalyzeResultsInput.model_validate(_request(raw)), state_no_sim
    )
    without_run_code = frozenset(state_no_sim.tool_dispatch) - {"run_code"}
    (failure,) = render_attached_analysis(stored, fields=None, served=without_run_code)["failures"]
    assert failure["message"].endswith(python_route_text(without_run_code))
    assert "run_code" not in failure["message"]


async def test_plot_waveform_gives_the_same_route(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    result = await call_tool(
        fake_request_context(state_no_sim),
        call_tool_params(
            "plot_waveform", {"path": str(raw), "signals": [_EXPRESSION], "open": False}
        ),
    )
    assert result.is_error
    data = result.structured_content
    assert data is not None
    assert data["error"].endswith(python_route_text(state_no_sim.tool_dispatch))
