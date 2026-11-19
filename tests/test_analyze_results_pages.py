"""A page of one list holds that list, and what identifies the page.

A cut reply's lists are read on one page at a time, from the stored answer.
A page holds the entries of its list, the ``page`` that says which list it
is of and where it starts, the ``stored_answer`` it reads and the cursor
for what follows, and nothing the first reply already gave: so its entries
take most of its size, and reading a list takes few calls.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.presenter import resolve_pointer
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.state import SessionState
from tests.test_analyze_results_through_the_presenter import _analyze, _job

#: The least share of a page's size its own list's entries take, when the
#: page holds more than one entry.
PAGE_SHARE = 0.5


async def _read_every_list(state: SessionState, first: dict[str, Any]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for pointer, entry in first.get("cut", {}).items():
        if "next" not in entry:
            continue
        cursor, calls, read, share = entry["next"], 1, entry["returned"], 1.0
        size = estimate_tokens(resolve_pointer(first, pointer))
        while cursor:
            page = await _analyze(
                state, {"continue": {"result_set_id": first["result_set_id"], "cursor": cursor}}
            )
            calls += 1
            entries = resolve_pointer(page, pointer)
            read += len(entries)
            size += estimate_tokens(entries)
            if len(entries) > 1:
                share = min(share, estimate_tokens(entries) / estimate_tokens(page))
            cursor = page["cut"][pointer].get("next")
        found[pointer] = {
            "calls": calls,
            "read": read,
            "total": entry["total"],
            "share": share,
            "size": size,
        }
    return found


def _within_bounds(found: dict[str, dict], target: int) -> None:
    for pointer, seen in found.items():
        assert seen["read"] == seen["total"], pointer
        assert seen["share"] >= PAGE_SHARE, (pointer, seen)
        # Each page after the first carries at least PAGE_SHARE of the
        # target in entries, or the rest of the list.
        bound = 1 + math.ceil(seen["size"] / (PAGE_SHARE * target)) + 1
        assert seen["calls"] <= bound, (pointer, seen, bound)


@pytest.mark.asyncio
async def test_a_grouped_spec_with_rows_reads_on_in_few_pages(
    state_no_sim: SessionState, work_dir: Path
):
    job = _job(state_no_sim, work_dir, 60)
    recipe: dict[str, Any] = {"metric": "value", "expr": "V(out)", "spec": {"max": 0.1}}
    first = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job, "label": "dut"}],
            "recipes": [
                {**recipe, "key": "a", "at": "100u"},
                {**recipe, "key": "b", "at": "900u"},
            ],
            "group_by": ["R"],
            "include": {"per_run": {"limit": 100}, "outliers": True, "provenance": True},
            "budget": 1500,
        },
    )
    _within_bounds(await _read_every_list(state_no_sim, first), 1500)


@pytest.mark.asyncio
async def test_three_value_recipes_over_256_cases_read_on_in_few_pages(
    state_no_sim: SessionState, work_dir: Path
):
    job = _job(state_no_sim, work_dir, 256)
    first = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job, "label": "dut"}],
            "recipes": [
                {"key": f"v_{at}", "metric": "value", "expr": "V(out)", "at": at}
                for at in ("100u", "500u", "900u")
            ],
        },
    )
    _within_bounds(await _read_every_list(state_no_sim, first), state_no_sim.config.default_budget)
