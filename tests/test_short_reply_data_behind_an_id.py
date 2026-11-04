"""A short reply, the data behind an id.

Design principle "A short reply, the data behind an id" (``docs/DESIGN.md``): a
reply carries a summary that is true on its own, and the bulk stays on disk,
recorded per job and case and addressed by id and path, so an old run reloads
without simulating again.

Every call goes through the server's own dispatch (``server.call_tool``).
The reply checker in ``tests/conftest.py`` holds every reply in the suite to
the rest of the principle: a list cut short gives the way to read the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    FakeSim,
    call_tool_params,
    fake_request_context,
    recorded_fixture_simulator,
)

pytestmark = pytest.mark.asyncio

_RECIPE: dict[str, Any] = {"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}


async def _call(state: SessionState, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await call_tool(fake_request_context(state), call_tool_params(tool, arguments))
    data = result.structured_content
    assert data is not None, result.content
    return data


def _deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text(
        "V1 in 0 PULSE(0 1 0 1n 1n 1m 2m)\nR1 in out 1k\nC1 out 0 100n\n.tran 1m\n.end\n",
        encoding="utf-8",
        newline="\n",
    )
    return deck


def _session(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"fake": FakeSim})


async def test_a_job_reloads_by_id_in_a_new_session_without_simulating_again(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    launches = recorded_fixture_simulator(monkeypatch)
    first = _session(config)
    receipt = await _call(
        first,
        "run_experiments",
        request_id="kept-on-disk",
        circuits=[{"path": str(_deck(work_dir)), "id": "rc"}],
        analyze={"recipes": [_RECIPE]},
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    measured = receipt["analysis"]["result"]["results"]["vout"]["values"][0]["value"]
    assert len(launches) == 1
    await first.shutdown()
    del first

    later = _session(config)
    assert receipt["job_id"] not in later.all_jobs
    status = await _call(later, "jobs", action="status", job_id=receipt["job_id"])
    rows = (await _call(later, "jobs", action="runs", job_id=receipt["job_id"]))["items"]
    read = await _call(
        later, "analyze_results", sources=[{"job_id": receipt["job_id"]}], recipes=[_RECIPE]
    )

    assert status["status"] == "completed", status
    # The receipt's paths are the files themselves.
    (row,) = rows
    raw, log = Path(row["raw"]), Path(row["log"])
    assert raw.read_bytes() == (FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes()  # noqa: ASYNC240
    assert log.read_bytes() == (FIXTURES_DIR / "ltspice_tran_rc.log").read_bytes()  # noqa: ASYNC240
    assert read["results"]["vout"]["values"][0]["value"] == measured
    assert len(launches) == 1, "reading the job by id ran the simulator again"
    await later.shutdown()


# ---------------------------------------------------------------------------
# Reply sizes
#
# The size, in characters, of a reply's structuredContent as compact JSON, with
# the working directory written as "W" so the bound is the same on every
# machine. A reply is read by the model in full on every call, so each pin is
# an upper bound: a field or a row that grows past it fails here, and the pin
# is raised deliberately in the change that earns it. A shrink needs no re-pin.
# ---------------------------------------------------------------------------


def _size(data: dict[str, Any], work_dir: Path) -> int:
    spellings = sorted({str(work_dir), str(work_dir.resolve())}, key=len, reverse=True)

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for spelling in spellings:
                value = value.replace(spelling, "W")
            return value
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return len(json.dumps(scrub(data), ensure_ascii=False, separators=(",", ":")))


#: A finished 10-case sweep's receipt, with no analysis attached: the job, its
#: source and completeness, and a row per case with its assignment and its raw
#: and log paths. The ten rows are 85% of it (3573 characters when pinned).
_SWEEP_RECEIPT_CHARS = 3700

#: analyze_results' reply for one value recipe over the same ten runs, grouped
#: by the swept part: the overall mean, one mean per group, and a digest per
#: run read (3218 characters when pinned).
_SWEEP_ANALYSIS_CHARS = 3350


async def test_a_sweeps_receipt_and_its_analysis_stay_short(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded_fixture_simulator(monkeypatch)
    values = [f"{k}k" for k in range(1, 11)]
    receipt = await _call(
        state_with_sim,
        "run_experiments",
        request_id="ten-cases",
        circuits=[{"path": str(_deck(work_dir)), "id": "rc"}],
        variations=[{"kind": "assign", "assign": {"R1": values}}],
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    assert receipt["completeness"]["produced"] == 10, receipt
    analysis = await _call(
        state_with_sim,
        "analyze_results",
        sources=[{"job_id": receipt["job_id"]}],
        recipes=[{**_RECIPE, "reduce": ["mean"]}],
        group_by=["R1"],
    )
    assert len(analysis["results"]["vout"]["groups"]) == 10, analysis

    sizes = {"receipt": _size(receipt, work_dir), "analysis": _size(analysis, work_dir)}
    assert sizes["receipt"] <= _SWEEP_RECEIPT_CHARS, sizes
    assert sizes["analysis"] <= _SWEEP_ANALYSIS_CHARS, sizes
