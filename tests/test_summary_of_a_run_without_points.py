"""The ``summary`` of a run whose results file holds no point: what its log says.

LTspice stops some runs before their first solution: XVII then writes a
results file whose plot declares no point (``ltspice17/log/err_timestep``),
and 26 an empty one (``ltspice26/log/err_source_loop`` and the others below).
The question is "what does this run's summary say", answered by
``metrics.summary`` on every route: by path, by job, and through the Python
API. Each gives the errors and conditions the log reader extracts, and says
that the results file holds no point; a run with no log beside it says that
no log was found; a results file with points is read as before.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import log_conditions, wsl
from ltspice_mcp.lib.log_parser import extract_log_diagnostics
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi, dispatched, terminal_experiment
from tests.test_log_condition_grid import REPORTED, _replaying, _staged

#: The recorded runs whose results file holds no point.
_NO_POINT = [
    ("ltspice17", "log/err_timestep"),
    ("ltspice26", "log/err_source_loop"),
    ("ltspice26", "log/err_inductor_loop"),
    ("ltspice26", "log/err_zero_resistance"),
]

_SUMMARY: dict[str, Any] = {
    "recipes": [{"key": "s", "metric": "summary"}],
    "include": {"per_run": {"limit": 10}, "fields": ["value"]},
}


def _value(reply: dict[str, Any]) -> dict[str, Any]:
    assert "items" in reply["results"]["s"].get("per_run", {}), reply.get("failures")
    (row,) = reply["results"]["s"]["per_run"]["items"]
    return row["value"]


def _state(work_dir: Path, build: str, case_id: str) -> SessionState:
    return SessionState.create(
        ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice"),
        available={"ltspice": _replaying(work_dir, build, case_id)},
    )


def _log_says(build: str, log: Path) -> tuple[list[Any], list[str]]:
    """The diagnostics the log reader extracts with the build's own rules,
    and the error lines the log reader finds in it."""
    extracted = extract_log_diagnostics(log)
    producer = Producer("ltspice", REPORTED[build], rec.manifest(build)["executable"]["name"])
    said = log_conditions.run_diagnostics(extracted["conditions"], producer)
    assert said.get("diagnostics"), said
    return said["diagnostics"], extracted["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id"), _NO_POINT, ids=[f"{b}/{c}" for b, c in _NO_POINT])
async def test_a_run_without_points_summarises_its_log_on_every_route(
    build: str, case_id: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    _, raw, log = _staged(build, case_id, work_dir / "path")
    assert rec.has(build, f"{case_id}.raw")
    diagnostics, errors = _log_says(build, log)
    assert errors
    state = _state(work_dir, build, case_id)
    api = SyncApi(state)
    by_path = [{"path": str(raw), "label": "p"}]
    routes = {
        "by path": _value(await dispatched(state, "analyze_results", sources=by_path, **_SUMMARY)),
        "api, by path": _value(
            await asyncio.to_thread(lambda: api.analyze_results(sources=by_path, **_SUMMARY))
        ),
    }
    run_deck = work_dir / "run" / Path(rec.CASES.case(case_id).source).name
    run_deck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rec.INPUTS / rec.CASES.case(case_id).source, run_deck)
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"no-point-{build}-{Path(case_id).name}",
            "circuits": [{"path": str(run_deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    by_job = [{"job_id": receipt["job_id"], "label": "j"}]
    reply = await dispatched(state, "analyze_results", sources=by_job, **_SUMMARY)
    routes["by job"] = _value(reply)
    routes["api, by job"] = _value(
        await asyncio.to_thread(lambda: api.analyze_results(sources=by_job, **_SUMMARY))
    )
    for route, value in routes.items():
        assert value.get("results_hold_no_point") is True, (route, value)
        assert "log_not_found" not in value, route
        assert value.get("diagnostics") == diagnostics, (route, value)
        assert "diagnostics_coverage" not in value, (route, value)
        # The conditions carry every error line; ``errors`` lists those the
        # simulator wrote the word on.
        assert set(value.get("errors", [])) <= set(errors), (route, value)
        assert "plots" not in value and "signals" not in value, (route, value)


@pytest.mark.asyncio
async def test_a_results_file_without_points_and_no_log_says_no_log_was_found(
    state_no_sim: SessionState, work_dir: Path
):
    raw = work_dir / "alone.raw"
    shutil.copyfile(rec.recorded("ltspice17", "log/err_timestep.raw"), raw)
    reply = await dispatched(
        state_no_sim, "analyze_results", sources=[{"path": str(raw), "label": "p"}], **_SUMMARY
    )
    value = _value(reply)
    assert value.get("results_hold_no_point") is True, value
    assert value.get("log_not_found") is True, value
    assert "diagnostics" not in value, value


@pytest.mark.asyncio
async def test_a_results_file_with_points_is_summarised_from_its_plots(
    state_no_sim: SessionState, work_dir: Path
):
    _, raw, _ = _staged("ltspice26", "deck/after_end", work_dir)
    value = _value(
        await dispatched(
            state_no_sim, "analyze_results", sources=[{"path": str(raw), "label": "p"}], **_SUMMARY
        )
    )
    assert "results_hold_no_point" not in value and "log_not_found" not in value, value
    assert value.get("point_count", 0) > 0 and value.get("sim_type"), value
