"""The size search, held to its rule over many lists and targets.

A reply is over its target only when every list it cut, held at one entry,
is over it together. And, for lists of whole entries (no series) at targets
at or above that floor, a larger target never gives a reply more than
:data:`MONOTONE_TOLERANCE` smaller. That second rule does not hold beyond
those cases: a series is redrawn at the most points that fit, so a larger
target can redraw it differently (eight waveform rows: 4245 estimated tokens
at 4500, 3897 at 4550), and below the floor the note that says a reply is
over its target adds its own size (2369 at 2100, 2137 at 2150).
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.presenter import present
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.store import Store

#: How far, in estimated tokens, a reply may shrink when its target rises:
#: the search cuts the largest list first, so a larger target can keep a
#: different list longer, whose cursors and notes differ by a few tokens.
MONOTONE_TOLERANCE = 40


def _schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "array", CUT: ITEMS, "items": {"type": "object"}} for name in names
        },
    }


def _answer(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    names = [f"list_{index}_{'k' * rng.randint(0, 50)}" for index in range(rng.randint(2, 8))]
    answer = {
        name: [
            {"entry": entry, "pad": "p" * rng.randint(1, 200)}
            for entry in range(rng.randint(1, 40))
        ]
        for name in names
    }
    return answer, _schema(names)


def _note(reply: dict[str, Any], cut: Any, stored: Any, paged: Any) -> None:
    """A note on the entries left out, as ``rows_not_shown`` is: it names a
    few of them, so its size moves with the counts kept, not with them."""
    named = {
        pointer: [
            f"left-{index}-{'n' * (index % 7) * 9}" for index in range(entry.returned, entry.total)
        ][:3]
        for pointer, entry in cut.items()
    }
    if named:
        reply["note"] = named


def _shown(answer: dict[str, Any], schema: dict[str, Any], target: int, tmp: Path) -> dict:
    # The bound reserved is a first estimate, below the note's largest size,
    # as the presenter allows (``_fit``): the whole reply at a count decides.
    return present(
        answer,
        schema,
        target=target,
        tool="t",
        snapshot={},
        store=Store(tmp),
        ttl_hours=1,
        annotate=_note,
        apart={"/note": 30},
    ).reply


@pytest.mark.parametrize("seed", range(60))
def test_a_reply_is_over_its_target_only_when_its_floor_is(seed: int, tmp_path: Path):
    rng = random.Random(seed)
    answer, schema = _answer(rng)
    whole = estimate_tokens(answer)
    floor_reply = _shown(answer, schema, 1, tmp_path)
    floor = estimate_tokens(floor_reply)
    sizes = []
    for target in sorted(rng.sample(range(max(100, floor - 200), whole + 200), 25)):
        shown = _shown(answer, schema, target, tmp_path)
        size = estimate_tokens(shown)
        if floor <= target:
            assert size <= target, (target, size, floor)
            sizes.append(size)
    for smaller, larger in itertools.pairwise(sizes):
        assert larger >= smaller - MONOTONE_TOLERANCE, sizes
