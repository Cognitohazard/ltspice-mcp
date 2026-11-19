"""What a cut analysis reply says, held against the ways it went wrong.

Following a reply's ``next`` from page to page ends; a thinned waveform is
thinned for the rows a reply shows, and its ``cut`` names only those; the
Python API gets a receipt's attached analysis with every failure; a row an
observation names after a source drifted is the row it was made of; and the
part on rows left out names at most three kinds, whatever a recipe is keyed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import not_shown
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools._base import automatic_door
from tests.test_analyze_results_through_the_presenter import _analyze, _job


@pytest.mark.asyncio
async def test_following_next_from_page_to_page_ends_and_reads_every_list(
    state_no_sim: SessionState, work_dir: Path
):
    job = _job(state_no_sim, work_dir, 40)
    request = {
        "sources": [{"job_id": job, "label": "dut"}],
        "recipes": [
            {"key": "a", "metric": "value", "expr": "V(out)", "at": "100u"},
            {"key": "b", "metric": "value", "expr": "V(out)", "at": "900u"},
        ],
        "include": {"per_run": {"limit": 100}},
        "budget": 1500,
    }
    reply = await _analyze(state_no_sim, request)
    seen: dict[str, list[Any]] = {
        key: list(reply["results"][key]["per_run"].get("items", [])) for key in ("a", "b")
    }
    for _ in range(200):
        if "next" not in reply:
            break
        reply = await _analyze(state_no_sim, {"continue": reply["next"]})
        for key in ("a", "b"):
            seen[key].extend(reply["results"][key]["per_run"].get("items", []))
    else:
        pytest.fail("following next never ended")
    whole = await asyncio.to_thread(
        lambda: (
            __import__("tests.conftest", fromlist=["SyncApi"])
            .SyncApi(state_no_sim)
            .analyze_results(**{key: value for key, value in request.items() if key != "budget"})
        )
    )
    for key in ("a", "b"):
        assert seen[key] == whole["results"][key]["per_run"]["items"], key


@pytest.mark.asyncio
async def test_a_series_is_thinned_for_the_rows_the_reply_shows(
    state_no_sim: SessionState, work_dir: Path
):
    request = {
        "recipes": [{"key": "w", "metric": "waveform", "signals": ["V(out)"], "max_points": 400}],
        "include": {"per_run": {"limit": 100}},
        "budget": 3000,
    }
    few = await _analyze(
        state_no_sim,
        {"sources": [{"job_id": _job(state_no_sim, work_dir, 3, "few"), "label": "d"}], **request},
    )
    many = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": _job(state_no_sim, work_dir, 30, "many"), "label": "d"}],
            **request,
        },
    )
    shown = many["results"]["w"]["per_run"]["items"]
    # Every cut entry names a list the reply holds.
    for pointer in many["cut"]:
        if pointer.endswith("/value/x"):
            index = int(pointer.split("/")[-3])
            assert index < len(shown), pointer
    # Rows the reply leaves out do not thin the rows it shows below what the
    # same rows get when nothing is left out.
    if len(shown) <= len(few["results"]["w"]["per_run"]["items"]):
        assert (
            len(shown[0]["value"]["x"])
            >= len(few["results"]["w"]["per_run"]["items"][0]["value"]["x"]) * 0.8
        )


def test_the_python_api_gets_an_attached_analysis_with_every_failure():
    stored = analyze_mod.analysis_snapshot.envelope(
        {
            "answer": {
                "results": {},
                "observations": [],
                "failures": [
                    {"code": "x", "stage": "s", "where": f"run-{index}", "message": "same"}
                    for index in range(3)
                ],
            },
            "snapshot": {},
            "request": {},
        }
    )
    with automatic_door():
        served = analyze_mod.attached_served(frozenset({"run_code"}))
    shown = analyze_mod.render_attached_analysis(stored, served=served)
    assert len(shown["failures"]) == 3


def test_the_part_on_rows_left_out_names_at_most_three_kinds():
    facts = not_shown.Facts(
        (
            (0, {"case_id": "a"}, ("k1", "k2")),
            (1, {"case_id": "b"}, ("k3", "k4")),
            (2, {"case_id": "c"}, ("k5",)),
        ),
        frozenset(),
    )
    part = not_shown.not_shown(
        {"g": not_shown.LeftOut(facts, 0, lambda _index: None)},
        bound=10_000,
        field="f",
        entries="rows",
    )
    named = [kind for row in part["g"]["rows"] for kind in row["kinds"]]
    assert named == ["k1", "k2", "k3"]


def test_a_recipe_keyed_like_the_folded_group_is_not_folded_into_it():
    facts = not_shown.Facts(
        tuple((index, {"case_id": f"c{index}"}, (f"k{index}",)) for index in range(40)),
        frozenset(),
    )
    groups = {
        key: not_shown.LeftOut(facts, 0, lambda _index: None) for key in ("*", "a", "b", "c")
    }
    part = not_shown.not_shown(groups, bound=300, field="f", entries="rows")
    assert part["*"]["counts"] == {f"k{index}": 1 for index in range(40)}
    assert "groups" not in part["*"]
