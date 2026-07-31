"""Caller-set response budget — a deterministic four-rung degradation ladder.

A read tool may carry an optional ``budget`` in estimated tokens. Absent, none
of this runs and the response is byte-for-byte what it has always been. Present,
the tool renders its envelope, the serializer measures it, and while it is over
the tool re-renders one rung further down a fixed ladder:

0. ``trim``   — suppress presentation blocks named on an explicit per-tool
   allowlist: optional keys are REMOVED when empty, required keys are EMPTIED
   in place. An allowlist rather than a predicate because a rung that exempts
   content is the one place a checker silently loses coverage.
1. ``answer`` — revoke the caller's payload-growing opt-ins so the response
   falls back to the answer channel it would have had by default.
2. ``columnar`` — render row surfaces as a column list plus rows of values,
   dropping the per-row key repetition. Lossless, so it applies to the answer
   rows too.
3. ``shrink`` — shrink the effective list limits BEFORE assembly, so a cursor
   is minted against what was actually returned. Never post-hoc truncation of
   an assembled page: a per_run cursor commits during evaluation, and trimming
   rows afterwards would point it past rows the caller never saw.

Three rules hold at every rung:

- **Facts survive any budget.** failures, observations, warnings, completeness
  and spec verdicts are never trimmed, empty or not — an empty ``failures`` IS
  the answer to "did anything fail". A budget squeezes presentation only.
- **Schema-safe by construction.** No rung deletes a required key; a required
  presentation block is emptied, never removed, so every emission still
  validates against the tool's declared output schema.
- **Termination rests on rung finiteness.** The ladder ends after rung 3
  whether or not the budget was met. A budget smaller than the facts floor
  returns the floor plus an observation saying so — over budget by honesty.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The pin convention for estimating tokens from a payload: compact-JSON
# characters divided by four. It is the same arithmetic the fleet transcripts
# were measured with (response chars, priced as model input), so a budget
# quoted here means the same thing as the numbers those campaigns report.
CHARS_PER_TOKEN = 4

# Floor on a caller-set budget. Below it the ladder has nothing to negotiate:
# the fact channels of a real analysis response (a completeness line, one
# failure record, one observation) already cost more than this, so a smaller
# number could only ever produce the floor plus the shortfall observation.
BUDGET_MIN_TOKENS = 500

# Rung levels, mildest first. RUNG_NONE is the undegraded response — a budget
# that is already met changes nothing at all.
RUNG_NONE = -1
RUNG_TRIM = 0
RUNG_ANSWER = 1
RUNG_COLUMNAR = 2
RUNG_SHRINK = 3

LADDER: tuple[int, ...] = (RUNG_NONE, RUNG_TRIM, RUNG_ANSWER, RUNG_COLUMNAR, RUNG_SHRINK)

_RUNG_NAMES: dict[int, str] = {
    RUNG_NONE: "none",
    RUNG_TRIM: "trim",
    RUNG_ANSWER: "answer",
    RUNG_COLUMNAR: "columnar",
    RUNG_SHRINK: "shrink",
}

BUDGET_DESCRIPTION_HEAD = (
    "Cap this response at roughly this many tokens (compact characters / 4), "
    "minimum 500. Omitted, the response is exactly what it would be without "
    "this argument. Set, the server measures the assembled response and, while "
    "it is over, degrades presentation down a fixed ladder: empty blocks and "
    "the identity echo, then your detail opt-ins, then columnar rows, then "
    "smaller pages (with cursors minted against the smaller page, so paging "
    "still walks every row). Facts are never cut at any budget — failures, "
    "observations, warnings and completeness always come back whole, and a "
    "budget too small for them returns them anyway and says so. The budget "
    "changes presentation only: it is not part of a result's identity, so the "
    "same request at two budgets shares one result set and one set of cursors."
)


def budget_description(cut_first: str) -> str:
    """The shared budget prose plus the tool-specific note on what gives first."""
    return f"{BUDGET_DESCRIPTION_HEAD} On this tool the first thing to give is {cut_first}."


@dataclass(frozen=True)
class Rung:
    """One step of the ladder, handed to a tool's renderer.

    ``measured`` is the token estimate of the previous (milder) rung, so the
    shrink rung can size a page against a real measurement instead of a guess.
    """

    level: int
    budget: int
    measured: int

    @property
    def trim(self) -> bool:
        return self.level >= RUNG_TRIM

    @property
    def answer_channel(self) -> bool:
        return self.level >= RUNG_ANSWER

    @property
    def columnar(self) -> bool:
        return self.level >= RUNG_COLUMNAR

    @property
    def shrink(self) -> bool:
        return self.level >= RUNG_SHRINK

    @property
    def name(self) -> str:
        return _RUNG_NAMES[self.level]

    @property
    def body_budget(self) -> int:
        """The budget less the room the budget notes themselves will take."""
        return self.budget - NOTE_RESERVE_TOKENS


def estimate_tokens(payload: Any) -> int:
    """Token estimate for ``payload`` — the serializer itself, measure-then-degrade.

    Serializes exactly as the transport does (compact separators, no ASCII
    escaping beyond JSON's own), with ``default=str`` so an unserializable leaf
    is measured at its string length rather than raising mid-ladder.

    One thing it cannot see: the non-finite substitution note the response
    builder appends when a payload carries NaN/Inf. That note lives on the
    tools layer, which this module sits below, and it is a fact channel that a
    budget would not cut anyway — so a degenerate result is a few tens of
    tokens heavier than measured here.
    """
    text = json.dumps(payload, separators=(",", ":"), default=str)
    return len(text) // CHARS_PER_TOKEN


def fits(payload: Any, budget: int) -> bool:
    return estimate_tokens(payload) <= budget


# --------------------------------------------------------------------------
# Rung 0 primitives — remove optional, empty required
# --------------------------------------------------------------------------


def remove_when_empty(container: dict[str, Any], key: str) -> None:
    """Drop an OPTIONAL presentation key whose value carries nothing.

    Only when empty: a non-empty optional block is content, and content is the
    business of the rungs above this one.
    """
    if key in container and not container[key]:
        del container[key]


def empty_page(page: dict[str, Any]) -> None:
    """Empty a page object's rows in place, keeping every required counter.

    ``total`` still states how many rows exist and ``next_cursor`` still points
    at them, so an emptied page is a handle, not a loss.
    """
    page["items"] = []
    page["returned"] = 0


def empty_required(container: dict[str, Any], key: str) -> None:
    """Empty a REQUIRED presentation key in place — never delete it.

    Deleting it would break the tool's own output schema under
    ``additionalProperties: false``; emptying keeps every emission valid.
    """
    value = container.get(key)
    if isinstance(value, list):
        container[key] = []
    elif isinstance(value, dict):
        container[key] = {}


# --------------------------------------------------------------------------
# Rung 2 primitives — columnar row surfaces
# --------------------------------------------------------------------------

COLUMNS_SUFFIX = "_columns"


def columnar_key(rows_key: str) -> str:
    """The sibling key naming the columns of ``rows_key``."""
    return f"{rows_key}{COLUMNS_SUFFIX}"


def columnarize(container: dict[str, Any], rows_key: str) -> bool:
    """Render ``container[rows_key]`` as columns + rows of values, in place.

    Returns whether the rendering happened. It is skipped unless every row is a
    dict with the SAME key set: a null-filled column cannot distinguish a key
    that was absent from one whose value was null, and the columnar form has to
    be lossless to be a presentation change rather than a data change.
    """
    rows = container.get(rows_key)
    if not isinstance(rows, list) or len(rows) < 2:
        # One row saves nothing (a column list the same size as the row it
        # describes) and zero rows have no keys to name.
        return False
    if not all(isinstance(row, dict) for row in rows):
        return False
    columns = list(rows[0])
    key_set = set(columns)
    if any(set(row) != key_set for row in rows[1:]):
        return False
    container[rows_key] = [[row[column] for column in columns] for row in rows]
    container[columnar_key(rows_key)] = columns
    return True


COLUMNAR_ROWS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "Column names for this response's columnar rows, in row order. Present "
        "only when a 'budget' argument made the rows render columnar; each row "
        "is then an array of values positionally matching this list."
    ),
    "items": {"type": "string"},
}


# --------------------------------------------------------------------------
# Rung 3 primitive — size a page against a measurement
# --------------------------------------------------------------------------


def fit_limit(current: int, rows: Sequence[Any], rung: Rung) -> int:
    """The row limit whose page is expected to fit, from the measured envelope.

    Splits the previous rung's measurement into the fixed envelope and the rows
    it carried, then divides what the budget leaves by the per-row cost. Never
    grows the caller's limit and never goes below one row: a page of one row
    still carries a cursor, which is the route back to the rest.
    """
    shown = len(rows)
    if shown <= 0 or current <= 1:
        return current
    row_tokens = estimate_tokens(list(rows))
    fixed = max(0, rung.measured - row_tokens)
    per_row = max(1, row_tokens // shown)
    affordable = (rung.body_budget - fixed) // per_row
    return max(1, min(current, int(affordable)))


# --------------------------------------------------------------------------
# Observations — what was cut, and the route back
# --------------------------------------------------------------------------

# One kind for both codes: a budget cut is a fact about how the response was
# presented, not about whether the data behind it can be trusted.
OBSERVATION_KIND = "presentation"


def truncated_observation(rung: Rung, estimate: int, cut: str, route: str) -> dict[str, Any]:
    """The fact that a budget degraded this response, and how to get it back."""
    return {
        "code": "budget_truncated",
        "kind": OBSERVATION_KIND,
        "detail": (
            f"budget={rung.budget} est. tokens degraded this response to rung "
            f"{rung.level} ({rung.name}), leaving an estimated {estimate} tokens "
            f"before this note: {cut} "
            f"Facts (failures, observations, warnings, completeness) were not cut. {route}"
        ),
    }


def not_met_observation(rung: Rung, estimate: int) -> dict[str, Any]:
    """The fact that the floor is bigger than the budget — stated, not hidden."""
    return {
        "code": "budget_not_met",
        "kind": OBSERVATION_KIND,
        "detail": (
            f"budget={rung.budget} est. tokens could not be met: the fully degraded "
            f"response is an estimated {estimate} tokens, and what remains is the "
            "fact floor (failures, observations, warnings, completeness, verdicts) "
            "plus the handles to reach the rest. Facts are never cut to fit a "
            "budget, so this response is over it by honesty. Narrow the request "
            "itself — fewer sources, recipes or queries per call."
        ),
    }


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


# What the "this was degraded" note above costs, measured from the note itself
# at its widest. The ladder holds it back so a response that says it was
# degraded still fits the cap the caller asked for — a note that pushed the
# answer over the budget would make the budget a number the server quotes and
# then breaks. The unmet-budget note needs no reserve: it is only ever emitted
# on a response that is already over, and says exactly that.
NOTE_RESERVE_TOKENS = estimate_tokens(
    [
        truncated_observation(
            Rung(level=RUNG_SHRINK, budget=999_999, measured=999_999),
            999_999,
            cut="x" * 160,
            route="x" * 160,
        )
    ]
)


@dataclass(frozen=True)
class Negotiated:
    data: dict[str, Any]
    rung: Rung
    estimate: int

    @property
    def degraded(self) -> bool:
        return self.rung.level > RUNG_NONE

    @property
    def met(self) -> bool:
        return self.estimate <= self.rung.budget - NOTE_RESERVE_TOKENS


async def negotiate(
    budget: int,
    render: Callable[[Rung], Awaitable[dict[str, Any]]],
) -> Negotiated:
    """Render at the mildest rung of the ladder that fits inside ``budget``.

    ``render`` builds the whole envelope for a rung; it is called at most once
    per rung, in order, and the first result that fits is returned. If none
    fits, the last rung's response stands — the caller is expected to attach
    :func:`not_met_observation` to it.

    Fit is judged against the budget less :data:`NOTE_RESERVE_TOKENS`, the room
    the caller's own budget notes will take once appended.
    """
    body_budget = budget - NOTE_RESERVE_TOKENS
    measured = 0
    data: dict[str, Any] = {}
    rung = Rung(level=RUNG_NONE, budget=budget, measured=0)
    for level in LADDER:
        rung = Rung(level=level, budget=budget, measured=measured)
        data = await render(rung)
        measured = estimate_tokens(data)
        if measured <= body_budget:
            break
    return Negotiated(data=data, rung=rung, estimate=measured)
