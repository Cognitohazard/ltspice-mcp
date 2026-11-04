"""Each task agents reported doing in one call still takes one call.

Design principle "One call per intent" (``docs/DESIGN.md``): the caller states
what it wants and the server does the steps in between. When a task takes a
sequence of calls, the fix is an op or an option, not a documented sequence.

Every call here goes through the server's own dispatch (``server.call_tool``)
by ``Client``, which counts them, the way a client reaches the server. Each
scenario asserts that its task completes in exactly one call and checks the
task's real outcome: the file beside the sheet, the comparison's verdict, the
replayed job, the reduced number. A count may only go down.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    call_tool_params,
    fake_request_context,
    fake_simulator,
    make_experiment_job,
    recorded_fixture_simulator,
    stage_recorded_fixture,
)

pytestmark = pytest.mark.asyncio


class Client:
    """The server as a client reaches it: by ``server.call_tool``, counted."""

    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.calls: list[str] = []

    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        self.calls.append(tool)
        result = await call_tool(
            fake_request_context(self.state), call_tool_params(tool, arguments)
        )
        data = result.structured_content
        assert data is not None, result.content
        return data


# ---------------------------------------------------------------------------
# A case's results beside its sheet
# ---------------------------------------------------------------------------


async def test_a_cases_results_go_beside_its_sheet_in_one_call(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = work_dir / "amp.asc"
    shutil.copyfile(FIXTURES_DIR / "Draft1.asc", sheet)
    kept = work_dir / "runs" / "placed"
    kept.mkdir(parents=True)
    raw = stage_recorded_fixture(kept, "ltspice_tran_rc")
    job = make_experiment_job(state_no_sim, job_id="placed", raw=raw)
    job.cases[0].circuit_path = sheet
    job.sources[0].path = sheet
    job.sources[0].sha256 = sha256_file(sheet)
    client = Client(state_no_sim)

    placed = await client.call("jobs", action="place_beside_sheet", job_id="placed")

    assert client.calls == ["jobs"]
    assert placed["outcome"] == "complete", placed
    assert sheet.with_suffix(".raw").read_bytes() == raw.read_bytes()
    assert sheet.with_suffix(".log").read_bytes() == raw.with_suffix(".log").read_bytes()


# ---------------------------------------------------------------------------
# An edit checked against a reference
# ---------------------------------------------------------------------------

_DIVIDER_OPS: list[dict[str, Any]] = [
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 400, "y": 300},
    {"op": "add_component", "reference": "R2", "symbol": "res", "x": 700, "y": 300},
    {
        "op": "wire_pins",
        "from_pin": "R1.1",
        "to_pin": "R2.1",
        "waypoints": [{"x": 400, "y": 200}, {"x": 700, "y": 200}],
    },
    {"op": "add_net_label", "net": "vout", "pin": "R1.2"},
]
_DIVIDER_DECK = "Vin in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n"
_OTHER_DECK = "Vin in 0 5\nR1 in out 1k\nR2 out mid 2k\nR3 mid 0 3k\n.end\n"


@pytest.fixture
def ltspice_exports_the_divider(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """LTspice's export of the committed sheet, which a run here cannot make.

    Returns the sheets it was asked to export, so a test sees the export ran
    after the commit, on the edited sheet.
    """
    exported: list[Path] = []

    async def export(sheet: Path, _state: SessionState) -> str:
        exported.append(sheet)
        assert "SYMBOL res" in sheet.read_text(encoding="utf-8", errors="replace")  # noqa: ASYNC240
        return _DIVIDER_DECK

    monkeypatch.setattr(se, "_export_asc_to_netlist", export)
    return exported


async def test_an_edit_is_compared_with_its_reference_in_the_same_call(
    asc_state: SessionState, work_dir: Path, ltspice_exports_the_divider: list[Path]
):
    client = Client(asc_state)

    edited = await client.call(
        "edit_schematic",
        path="divider.asc",
        base="blank",
        ops=_DIVIDER_OPS,
        compare={"reference": _DIVIDER_DECK},
    )

    assert client.calls == ["edit_schematic"]
    assert edited["commit_state"] == "committed", edited
    assert edited["verification"]["equivalent"] is True, edited
    assert ltspice_exports_the_divider, "the comparison never exported the sheet"
    assert "SYMBOL res" in (work_dir / "divider.asc").read_text(encoding="utf-8")


async def test_a_mismatch_is_reported_in_the_same_call_without_undoing_the_edit(
    asc_state: SessionState, work_dir: Path, ltspice_exports_the_divider: list[Path]
):
    client = Client(asc_state)

    edited = await client.call(
        "edit_schematic",
        path="divider.asc",
        base="blank",
        ops=_DIVIDER_OPS,
        compare={"reference": _OTHER_DECK},
    )

    assert client.calls == ["edit_schematic"]
    assert edited["commit_state"] == "committed", edited
    assert edited["verification"]["equivalent"] is False, edited
    sheet = (work_dir / "divider.asc").read_text(encoding="utf-8")
    assert "SYMATTR InstName R2" in sheet, "the mismatch undid the edit"


# ---------------------------------------------------------------------------
# Two settings compared, and the comparison asked for again
# ---------------------------------------------------------------------------


def _deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text(
        "V1 in 0 PULSE(0 1 0 1n 1n 1m 2m)\nR1 in out 1k\nC1 out 0 100n\n.tran 1m\n.end\n",
        encoding="utf-8",
        newline="\n",
    )
    return deck


async def test_two_settings_are_run_and_measured_in_one_call(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded_fixture_simulator(monkeypatch)
    client = Client(state_with_sim)

    receipt = await client.call(
        "run_experiments",
        request_id="two-settings",
        circuits=[{"path": str(_deck(work_dir)), "id": "rc"}],
        variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
        analyze={
            "recipes": [
                {
                    "key": "vout",
                    "metric": "value",
                    "expr": "V(out)",
                    "at": "900u",
                    "reduce": ["mean"],
                }
            ],
            "group_by": ["R1"],
        },
        execution={"wait_s": LIVENESS_S},
    )

    assert client.calls == ["run_experiments"]
    assert receipt["status"] == "completed", receipt
    assert receipt["completeness"]["expanded"] == 2, receipt
    groups = receipt["analysis"]["result"]["results"]["vout"]["groups"]
    # Both cases replay one recorded raw, so the two means are the same number;
    # what this pins is that each setting is measured in the same reply.
    assert sorted(group["by"]["R1"] for group in groups) == ["1k", "2k"], groups
    for group in groups:
        ((reduced,),) = [group["reduced"]]
        assert reduced["stat"] == "mean" and isinstance(reduced["value"], float), group


async def test_repeating_a_request_id_answers_from_the_job_in_one_call(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    launches = fake_simulator(monkeypatch)
    arguments: dict[str, Any] = {
        "request_id": "asked-again",
        "circuits": [{"path": str(_deck(work_dir)), "id": "rc"}],
        "variations": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
        "execution": {"wait_s": LIVENESS_S},
    }
    first = await Client(state_with_sim).call("run_experiments", **arguments)
    assert first["status"] == "completed", first
    client = Client(state_with_sim)

    again = await client.call("run_experiments", **arguments)

    assert client.calls == ["run_experiments"]
    assert again["replayed"] is True, again
    assert again["job_id"] == first["job_id"]
    assert again["status"] == "completed", again
    assert len(launches) == 2, "the replay ran the cases again"


# ---------------------------------------------------------------------------
# A reduction in Python
# ---------------------------------------------------------------------------


async def test_a_jobs_results_are_reduced_in_python_in_one_call(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The snippet reads the job by id in the worker process, from the record
    on disk, and reduces it there; the reply carries the number."""
    monkeypatch.setenv("LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION", "1")
    recorded_fixture_simulator(monkeypatch)
    receipt = await Client(state_with_sim).call(
        "run_experiments",
        request_id="to-reduce",
        circuits=[{"path": str(_deck(work_dir)), "id": "rc"}],
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    expected = (FIXTURES_DIR / "ltspice_tran_rc.raw").stat().st_size
    client = Client(state_with_sim)
    try:
        reduced = await client.call(
            "run_code",
            code=(
                f"r = api.load_raw(job_id={receipt['job_id']!r})\n"
                "v = r.trace('V(out)')\n"
                "import os\n"
                "(os.path.getsize(r.source), round(float(np.max(v)), 9), len(v) > 1)"
            ),
            timeout_s=LIVENESS_S,
        )
    finally:
        await state_with_sim.shutdown()

    assert client.calls == ["run_code"]
    assert reduced["status"] == "ok", reduced
    size, peak, sampled = eval(reduced["result"], {})
    assert size == expected
    assert sampled is True
    assert 0.0 < peak <= 1.0 + 1e-9, reduced
