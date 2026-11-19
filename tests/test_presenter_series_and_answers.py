"""Two powers of the presenter: thinning a series, and several answers
under one target.

A series is samples a schema declares under one name, a waveform's axis and
its values: when a reply is too long for them, the presenter redraws them at
the most points that fit, by a function the tool supplies, aligned and never
below its floor. Several answers under one target are one reply holding
answers of different tools, each at its own root: one size search runs over
the lists of all of them, so a reply that fits is whole, and each answer
carries its own ``cut`` and ``stored_answer``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.presenter import (
    Answer,
    Series,
    open_page,
    present,
    present_many,
    present_page,
    resolve_pointer,
)
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS, NEVER, series
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.store import Store
from ltspice_mcp.tools.analyze import SERIES, THIN_FLOOR_POINTS

_SAMPLES = {"type": "array", "items": {"type": "number"}}

#: Rows whose value holds a waveform: an axis and one list per signal.
_WAVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            CUT: ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "object",
                        "properties": {
                            "x": {**_SAMPLES, CUT: series("w")},
                            "series": {
                                "type": "array",
                                CUT: series("w"),
                                "items": {
                                    "type": "object",
                                    "additionalProperties": {**_SAMPLES, CUT: NEVER},
                                },
                            },
                        },
                    }
                },
            },
        }
    },
}


def _wave(rng: random.Random, points: int, signals: int) -> dict[str, Any]:
    x = sorted(rng.uniform(0, 1) for _ in range(points))
    return {
        "format": "inline",
        "x": x,
        "series": [
            {"signal": f"V(n{index})", "y": [rng.uniform(-1, 1) for _ in x]}
            for index in range(signals)
        ],
        "points_returned": points,
        "points_total": points,
    }


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path)


@pytest.mark.parametrize("seed", range(30))
def test_a_thinned_series_stays_aligned_and_never_falls_below_its_floor(seed: int, tmp_path: Path):
    rng = random.Random(seed)
    answer = {
        "rows": [
            {"value": _wave(rng, rng.randint(40, 400), rng.randint(1, 3))}
            for _ in range(rng.randint(1, 4))
        ]
    }
    target = rng.randint(200, estimate_tokens(answer))
    shown = present(
        answer,
        _WAVE_SCHEMA,
        target=target,
        tool="t",
        snapshot={},
        store=_store(tmp_path),
        ttl_hours=1,
        series=SERIES,
    ).reply
    for index, row in enumerate(shown["rows"]):
        whole = answer["rows"][index]["value"]
        value = row["value"]
        assert all(len(entry["y"]) == len(value["x"]) for entry in value["series"])
        assert len(value["x"]) >= min(THIN_FLOOR_POINTS, len(whole["x"]))
        assert value["points_returned"] == len(value["x"])
        if len(value["x"]) < len(whole["x"]):
            entry = shown["cut"][f"/rows/{index}/value/x"]
            assert (entry["total"], entry["returned"]) == (len(whole["x"]), len(value["x"]))
            assert "next" not in entry
            # Each bucket's extremes are kept.
            for before, after in zip(whole["series"], value["series"], strict=True):
                assert min(after["y"]) == min(before["y"])
                assert max(after["y"]) == max(before["y"])


def test_a_series_that_fits_is_not_thinned(tmp_path: Path):
    answer = {"rows": [{"value": _wave(random.Random(1), 50, 1)}]}
    shown = present(
        answer,
        _WAVE_SCHEMA,
        target=10**6,
        tool="t",
        snapshot={},
        store=_store(tmp_path),
        ttl_hours=1,
        series=Series(lambda holder, points: pytest.fail("redrawn")),
    )
    assert shown.reply is answer


# ---------------------------------------------------------------------------
# Several answers under one target
# ---------------------------------------------------------------------------

_INNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", CUT: ITEMS, "items": {"type": "object"}}},
}

_OUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {"type": "array", CUT: ITEMS, "items": {"type": "object"}},
        "inner": _INNER_SCHEMA,
    },
}


def _two_answers(rng: random.Random) -> dict[str, Any]:
    return {
        "rows": [
            {"row": index, "pad": "r" * rng.randint(5, 60)} for index in range(rng.randint(2, 30))
        ],
        "inner": {
            "items": [
                {"item": index, "pad": "i" * rng.randint(5, 60)}
                for index in range(rng.randint(2, 30))
            ]
        },
    }


def _answers() -> list[Answer]:
    return [
        Answer("", _OUTER_SCHEMA, "outer_tool", {"outer": 1}),
        Answer("/inner", _INNER_SCHEMA, "inner_tool", {"inner": 1}),
    ]


def _read_every_page(
    store: Store, first: dict[str, Any], list_pointer: str, tool: str, schema: dict[str, Any]
) -> list[Any]:
    rows = list(resolve_pointer(first, list_pointer))
    entry = first["cut"][list_pointer]
    while "next" in entry:
        stored, pointer, offset = open_page(entry["next"], schema, tool=tool, store=store, gone="")
        page = present_page(stored, pointer, offset, schema, target=None, cap=3).reply
        rows.extend(resolve_pointer(page, pointer))
        entry = page["cut"][pointer]
    return rows


@pytest.mark.parametrize("seed", range(40))
def test_several_answers_share_one_target(seed: int, tmp_path: Path):
    rng = random.Random(seed)
    answer = _two_answers(rng)
    store = _store(tmp_path)
    whole = estimate_tokens(answer)

    # A reply that fits goes out whole: no answer is cut while room remains.
    fits = present_many(answer, _answers(), target=whole + 50, store=store, ttl_hours=1)
    assert fits.reply is answer
    assert not (store.results_dir.exists() and list(store.results_dir.glob("sa_*.json")))

    target = rng.randint(min(150, whole), whole)
    shown = present_many(answer, _answers(), target=target, store=store, ttl_hours=1).reply
    outer_cut = shown.get("cut", {})
    inner_cut = shown["inner"].get("cut", {})
    # Each answer names its own lists, relative to its root, and its own copy.
    assert set(outer_cut) <= {"/rows"}
    assert set(inner_cut) <= {"/items"}
    if outer_cut:
        assert shown["stored_answer"]["id"] != shown["inner"].get("stored_answer", {}).get("id")
        assert (
            _read_every_page(store, shown, "/rows", "outer_tool", _OUTER_SCHEMA)
            == (answer["rows"])
        )
    if inner_cut:
        assert (
            _read_every_page(store, shown["inner"], "/items", "inner_tool", _INNER_SCHEMA)
            == (answer["inner"]["items"])
        )
    cuts = [*outer_cut.values(), *inner_cut.values()]
    assert all(entry["returned"] >= 1 for entry in cuts)
    # Over its target only when every list it cut holds one entry.
    if estimate_tokens(shown) > target:
        assert all(entry["returned"] == 1 for entry in cuts)


def test_an_inner_answers_cursor_is_read_by_its_own_tool(tmp_path: Path):
    answer = _two_answers(random.Random(3))
    store = _store(tmp_path)
    shown = present_many(answer, _answers(), target=150, store=store, ttl_hours=1).reply
    entry = shown["inner"]["cut"]["/items"]
    stored, pointer, offset = open_page(
        entry["next"], _INNER_SCHEMA, tool="inner_tool", store=store, gone=""
    )
    assert (stored.tool, pointer, offset) == ("inner_tool", "/items", entry["returned"])
    assert stored.answer == answer["inner"]
    assert stored.snapshot == {"inner": 1}
    with pytest.raises(Exception, match="not a cursor"):
        open_page(entry["next"], _OUTER_SCHEMA, tool="outer_tool", store=store, gone="")
