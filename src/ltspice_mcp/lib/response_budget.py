"""Caller-set response budget — a deterministic three-rung degradation ladder.

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
2. ``shrink`` — shrink the effective list limits BEFORE assembly, so a cursor
   is minted against what was actually returned. Never post-hoc truncation of
   an assembled page: a per_run cursor commits during evaluation, and trimming
   rows afterwards would point it past rows the caller never saw.

Four rules hold at every rung:

- **Facts survive any budget.** failures, observations, warnings, completeness
  and spec verdicts are never trimmed, empty or not — an empty ``failures`` IS
  the answer to "did anything fail". A budget squeezes presentation only.
- **Rows keep their shape.** A row is an object at every rung, keyed the same
  way whatever the budget: a tight budget returns fewer rows, never differently
  shaped ones. A rung that re-rendered rows as a column list plus arrays of
  values was removed for that reason — it made a caller branch on the shape of
  what came back.
- **Schema-safe by construction.** No rung deletes a required key; a required
  presentation block is emptied, never removed, so every emission still
  validates against the tool's declared output schema.
- **Termination rests on rung finiteness.** The ladder ends after rung 2
  whether or not the budget was met. A budget smaller than the facts floor
  returns the floor plus an observation saying so, and stays over the budget.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The convention for estimating tokens from a payload: compact-JSON characters
# divided by four. Response size is measured the same way everywhere in this
# project (response characters, priced as model input), so a budget quoted here
# means the same thing as any other response-size number.
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
RUNG_SHRINK = 2

_RUNG_NAMES: dict[int, str] = {
    RUNG_NONE: "none",
    RUNG_TRIM: "trim",
    RUNG_ANSWER: "answer",
    RUNG_SHRINK: "shrink",
}

# The walk order, derived rather than restated: a rung the driver climbs but
# cannot name would raise from inside the note it is writing.
LADDER: tuple[int, ...] = tuple(sorted(_RUNG_NAMES))

# One sentence per fact a caller needs to decide whether to set this: the unit,
# that presentation is all it touches, and that omitting it is not "no budget".
# The ladder's per-tool mechanics live in spice://guide — they are what a caller
# reads once, not what every session should pay for on the wire.
BUDGET_DESCRIPTION = (
    "Approximate response-token cap (compact characters / 4, minimum 500). "
    "Presentation degrades in a fixed order; facts — failures, observations, "
    "warnings, completeness — are never cut. Omitted, the server's own default "
    "budget applies and only strips empty blocks and the identity echo. "
    "Presentation only: it is not part of a result's identity. See spice://guide."
)


@dataclass(frozen=True)
class Rung:
    """One step of the ladder, handed to a tool's renderer.

    ``measured`` is the token estimate of the previous (milder) rung, so the
    shrink rung can size a page against a real measurement instead of a guess.
    """

    level: int
    budget: int
    measured: int
    #: Room held back for the budget's own epilogue. ``None`` means one copy of
    #: the truncation note; :class:`Notes` sets it for the tool that is
    #: negotiating, because a tool mirroring the note into a hint writes it twice.
    reserve: int | None = None

    @property
    def trim(self) -> bool:
        return self.level >= RUNG_TRIM

    @property
    def answer_channel(self) -> bool:
        return self.level >= RUNG_ANSWER

    @property
    def shrink(self) -> bool:
        return self.level >= RUNG_SHRINK

    @property
    def name(self) -> str:
        return _RUNG_NAMES[self.level]

    @property
    def body_budget(self) -> int:
        """The budget less the room the budget notes themselves will take."""
        return self.budget - (NOTE_RESERVE_TOKENS if self.reserve is None else self.reserve)


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


def apply_trim(
    container: dict[str, Any],
    *,
    remove: Sequence[str] = (),
    empty: Sequence[str] = (),
) -> None:
    """Rung 0 over one container, from a tool's declared key lists.

    ``remove`` names optional keys, dropped only when they carry nothing;
    ``empty`` names required keys, emptied in place. The lists are declared as
    module-level data by each tool rather than spelled inline here: a rung that
    exempts content is the one place a checker can silently lose coverage, so
    the allowlist has to be something a test can read.

    Idempotent, so the ladder may re-apply it to an already-trimmed envelope.
    """
    for key in remove:
        remove_when_empty(container, key)
    for key in empty:
        empty_required(container, key)


# --------------------------------------------------------------------------
# Rung 2 primitive — size a page against a measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RowMeasure:
    """One measurement of the rows a rung rendered, shared by every limit.

    A response with several row surfaces sizes them all against the same
    envelope, so the rows are measured once and each limit divides into that
    measurement rather than re-serializing the same rows per limit.
    """

    shown: int
    tokens: int

    @classmethod
    def of(cls, rows: Sequence[Any]) -> RowMeasure:
        return cls(shown=len(rows), tokens=estimate_tokens(list(rows)))

    def affordable(self, rung: Rung) -> int:
        """How many rows of the measured cost this budget leaves room for.

        Splits the previous rung's measurement into the fixed envelope and the
        rows it carried, then divides what the budget leaves by the per-row
        cost. Never below one row: a page of one row still carries a cursor,
        which is the route back to the rest.

        Every row surface a rung shrinks has to be in the measurement this was
        built from. A surface left out is charged to ``fixed`` — correct only
        while it genuinely cannot shrink, and an under-count of the fixed cost
        the moment it can.
        """
        fixed = max(0, rung.measured - self.tokens)
        per_row = max(1, self.tokens // self.shown) if self.shown > 0 else 1
        return max(1, (rung.body_budget - fixed) // per_row)

    def fit_limit(self, current: int, rung: Rung) -> int:
        """``current``, lowered to what :meth:`affordable` leaves room for.

        Never grows the caller's limit.
        """
        if self.shown <= 0 or current <= 1:
            return current
        return min(current, self.affordable(rung))


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
            f"response is an estimated {estimate} tokens. What remains is failures, "
            "observations, warnings, completeness and verdicts, plus the cursors and "
            "ids that reach the rest. Facts are not removed to fit a budget, so this "
            "response is over the budget instead. Narrow the request itself — fewer "
            "sources, recipes or queries per call."
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
class Notes:
    """A tool's budget epilogue: what it says, and what saying it costs.

    One value drives both ends of the ladder — :func:`negotiate` holds back the
    room the notes will need, :func:`attach_notes` writes them — so the two
    cannot disagree about how much room that is. That is the whole reason this
    is an object rather than two argument lists: the reserve is derived from the
    same ``hint_key`` that decides how many copies get written.
    """

    #: What this tool gave up, in its own terms.
    cut: str
    #: How the caller gets the rest back.
    route: str
    #: The guidance key to mirror the last note into, for tools whose
    #: structured-aware clients read guidance only from there.
    hint_key: str | None = None

    @property
    def reserve(self) -> int:
        """Room to hold back for the epilogue itself.

        Two copies when a hint key is set, because that is literally how many
        get written: the detail lands on ``observations`` AND again in the hint.
        Reserving one would let a response that reports its own truncation land
        over the cap the caller asked for — flagged as truncated, and silently
        wrong about having met the budget.
        """
        return NOTE_RESERVE_TOKENS * (2 if self.hint_key is not None else 1)


@dataclass(frozen=True)
class Negotiated:
    data: dict[str, Any]
    rung: Rung
    estimate: int
    #: Whether the ladder was stopped short of its last rung by policy rather
    #: than by fitting. A capped run that does not fit is the policy working,
    #: not a shortfall, so it reports no unmet-budget note.
    capped: bool = False

    @property
    def degraded(self) -> bool:
        return self.rung.level > RUNG_NONE

    @property
    def met(self) -> bool:
        return self.estimate <= self.rung.body_budget


async def negotiate(
    budget: int,
    render: Callable[[Rung], Awaitable[dict[str, Any]]],
    notes: Notes,
    *,
    max_rung: int = RUNG_SHRINK,
) -> Negotiated:
    """Render at the mildest rung of the ladder that fits inside ``budget``.

    ``render`` builds the whole envelope for a rung; it is called at most once
    per rung, in order, and the first result that fits is returned. If none
    fits, the last rung's response stands — the caller is expected to attach
    :func:`not_met_observation` to it, which :func:`attach_notes` does.

    ``max_rung`` stops the walk early. It exists for the server-side default
    budget, which may strip presentation the caller never asked to keep (rung
    0) but must not revoke opt-ins the caller DID ask for (rung 1 and below) —
    doing that unasked would answer a different question than the one asked.

    Fit is judged against the budget less ``notes.reserve``, the room this
    tool's own budget notes will take once appended.
    """
    measured = 0
    data: dict[str, Any] = {}
    rung = Rung(level=RUNG_NONE, budget=budget, measured=0, reserve=notes.reserve)
    met = False
    for level in LADDER:
        if level > max_rung:
            break
        rung = Rung(level=level, budget=budget, measured=measured, reserve=notes.reserve)
        data = await render(rung)
        measured = estimate_tokens(data)
        met = measured <= rung.body_budget
        if met:
            break
    return Negotiated(
        data=data,
        rung=rung,
        estimate=measured,
        capped=not met and max_rung < RUNG_SHRINK,
    )


def append_hint(data: dict[str, Any], detail: str, *, key: str = "hint") -> None:
    """Append caller guidance once without displacing an existing route."""
    existing = data.get(key)
    if not isinstance(existing, str) or not existing:
        data[key] = detail
    elif detail not in existing:
        data[key] = f"{existing} {detail}"


def attach_notes(result: Negotiated, notes: Notes) -> None:
    """Append the budget's own notes to a negotiated response, in place.

    One epilogue for every tool that negotiates. A degraded response says so and
    how to get the rest back; a response the ladder could not shrink into the
    budget says that too. Both go on ``observations`` by extension, never by
    assignment: a tool that already put facts there keeps them, which is the
    whole point of a channel a budget cannot cut.

    ``notes.hint_key`` mirrors the last note into the tool's guidance key — the
    second copy ``Notes.reserve`` already made room for.
    """
    written: list[dict[str, Any]] = []
    if result.degraded:
        written.append(
            truncated_observation(result.rung, result.estimate, cut=notes.cut, route=notes.route)
        )
    if not result.met and not result.capped:
        written.append(not_met_observation(result.rung, result.estimate))
    if not written:
        return
    data = result.data
    observations = data.setdefault("observations", [])
    observations.extend(written)
    if notes.hint_key is not None:
        append_hint(data, written[-1]["detail"], key=notes.hint_key)
