"""The case id and the order of cases are a stated contract (guide section
'variations', and ``docs/design/mcp_surface.md``): a caller may write a case id
or a run index before its job runs. Each statement fails a test here when it
stops being true, or is pinned by the test named beside it:

- A case id is ``<circuit id>-case-<number>``: the number counts the circuit's
  cases from 0, padded to four digits and wider past 9999 (here, and
  ``test_variations.py::test_native_samples_follow_assignments_and_keep_logical_indices``).
- The number counts per circuit (``test_variations.py::test_applies_to_routes_only_to_named_circuit``);
  ``run_index`` counts the whole job's cases from 0, circuits in the order given (here).
- Separate entries combine as a product, the last entry changing fastest
  (``test_variations.py::test_stable_order_across_assign_entries``); inside one
  grid entry the last target changes fastest
  (``test_variations.py::test_grid_is_cartesian_and_stable``); a zip entry pairs
  its targets (``test_variations.py::test_zip_is_lockstep``).
- The samples of a random or native entry change fastest of all, wherever the
  entry stands (here, with the random entry first; native last in
  ``test_variations.py::test_native_samples_follow_assignments_and_keep_logical_indices``).
- A resume keeps each case's id, run index and order: it copies the job's
  cases and expands nothing again
  (``test_resume_surface.py::test_resume_child_reuses_success_and_replays_after_source_deletion``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.variations import AssignVariation, RandomVariation, expand_variations
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import recorded_fixture_simulator
from tests.test_variations import _deck, _random


def test_a_case_number_is_padded_to_four_digits_and_wider_past_9999(tmp_path: Path):
    """The case cap ([simulation] max_experiment_cases) goes past 9999, so a
    number wider than four digits is one a job can hold."""
    assert ServerConfig().max_experiment_cases <= 9999
    assert ServerConfig(max_experiment_cases=10_001).max_experiment_cases == 10_001
    circuit = _deck(tmp_path / "dut.cir")
    sweep = AssignVariation(kind="assign", assign={"R1": [f"{i}" for i in range(10_001)]})
    cases = expand_variations([circuit], [sweep], max_cases=10_001)
    assert [cases[i].case_id for i in (0, 7, 9_999, 10_000)] == [
        "dut-case-0000",
        "dut-case-0007",
        "dut-case-9999",
        "dut-case-10000",
    ]


def test_samples_change_fastest_wherever_their_entry_stands(tmp_path: Path):
    circuit = _deck(tmp_path / "dut.cir")
    random: RandomVariation = _random(runs=2)
    sweep = AssignVariation(kind="assign", assign={"R1": ["1k", "2k"]})
    cases = expand_variations([circuit], [random, sweep])
    assert [(case.assignments["R1"], case.random_index) for case in cases] == [
        ("1k", 0),
        ("1k", 1),
        ("2k", 0),
        ("2k", 1),
    ]


def _payload(tmp_path: Path, request_id: str) -> dict[str, Any]:
    decks = []
    for name in ("amp", "filt"):
        deck = tmp_path / f"{name}.cir"
        deck.write_text(
            "* rc\nV1 in 0 1\nR1 in out {r}\nC1 out 0 1n\n.param r=1k\n.tran 1m\n.end\n"
        )
        decks.append({"path": str(deck), "id": name})
    return {
        "request_id": request_id,
        "circuits": decks,
        "variations": [{"kind": "assign", "assign": {"r": ["1k", "2k", "3k"]}}],
        "execution": {"wait_s": 60},
        "lint": "off",
    }


@pytest.mark.asyncio
async def test_run_index_counts_the_whole_job_and_the_case_number_each_circuit(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded_fixture_simulator(monkeypatch)
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(_payload(work_dir, "two-circuits")), state_with_sim
    )
    assert result.structured_content is not None
    job = state_with_sim.all_jobs[result.structured_content["job_id"]]
    assert [(case.case_id, case.run_index) for case in job.cases] == [
        ("amp-case-0000", 0),
        ("amp-case-0001", 1),
        ("amp-case-0002", 2),
        ("filt-case-0000", 3),
        ("filt-case-0001", 4),
        ("filt-case-0002", 5),
    ]
    assert [case.assignments["r"] for case in job.cases] == ["1k", "2k", "3k"] * 2
