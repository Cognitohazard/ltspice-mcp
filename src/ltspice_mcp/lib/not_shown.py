"""What a reply says of the entries it left out that carry a fact.

A reply cut short of a list of rows (a job's run rows, a recipe's rows) never
gives an all-clear for a row it does not show: the part this module builds
counts, per kind of fact, the entries left out that carry it, and names a few
of them, each with a cursor that starts at it. It is the one builder of that
part: ``tools/receipts.py`` (``runs_not_shown``) and ``tools/analyze.py``
(``rows_not_shown``) both call :func:`not_shown`, and neither groups or bounds
on its own (``tests/test_reply_cuts.py``).

The part holds one or more groups, each the entries left out of one list.
It has one stated overall bound, in estimated tokens:

- every kind keeps its count, in each group;
- at most :data:`NAMED` kinds are named over all groups, each by the first
  entry left out that carries it and that not every entry of its list
  carries, one entry for at most :data:`KINDS_PER_ENTRY` kinds; a kind every
  entry carries is given by its count alone;
- named entries come off from the last while the part is over its bound,
  then groups from the last are summed into one group under :data:`FOLDED`
  (starred again when a group is already named so).

The caller computes the bound from the worst part its vocabulary allows
(:func:`worst_bound`), and the presenter measures the whole reply at the
count it picks (``presenter._fit``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ltspice_mcp.lib.response_budget import estimate_tokens

#: How many kinds the part names an entry for, over all groups.
NAMED: Final = 3

#: How many kinds one named entry is named for.
KINDS_PER_ENTRY: Final = 2

#: The group the groups the bound left out are summed into.
FOLDED: Final = "*"

#: What a longer case id or count of a later page may add to the reserve a
#: caller makes for the part (``presenter._fit``'s ``apart``).
RESERVE_MARGIN: Final = 16


def log_kinds() -> list[str]:
    """The kinds of fact a run's log gives a row, whoever lists the row: each
    diagnostic code and each coverage value (``lib/log_conditions.py``)."""
    from ltspice_mcp.lib import log_conditions

    return [
        *(f"diagnostics.{code}" for code in sorted(log_conditions.diagnostic_codes())),
        *(f"diagnostics_coverage.{value}" for value in log_conditions.COVERAGE),
    ]


@dataclass(frozen=True)
class Facts:
    """The facts on every entry of one whole list, and the kinds every entry
    carries: ``entries`` holds ``(index, identity, kinds)`` for each entry
    that carries a fact, in list order, where ``identity`` names the entry
    (its case and run, say) as the part repeats it."""

    entries: tuple[tuple[int, Mapping[str, Any], tuple[str, ...]], ...]
    every: frozenset[str]

    @classmethod
    def of(
        cls,
        rows: Sequence[Any],
        kinds: Callable[[Any], Sequence[str]],
        identity: Callable[[int, Any], Mapping[str, Any]],
    ) -> Facts:
        """The facts of ``rows``: ``kinds`` gives one row's kinds of fact and
        ``identity`` what names it."""
        entries = tuple(
            (index, identity(index, row), tuple(found))
            for index, row in enumerate(rows)
            if (found := list(dict.fromkeys(kinds(row))))
        )
        every: frozenset[str] = frozenset()
        if rows and len(entries) == len(rows):
            every = frozenset.intersection(*(frozenset(found) for *_, found in entries))
        return cls(entries, every)


@dataclass(frozen=True)
class LeftOut:
    """One list's entries left out: its facts, the index the entries left out
    start at, and the cursor that starts at an entry (None: none)."""

    facts: Facts
    start: int
    cursor: Callable[[int], str | None]


def not_shown(
    groups: Mapping[str, LeftOut],
    *,
    bound: int,
    field: str,
    entries: str,
    keep: Callable[[str], bool] = lambda _kind: True,
) -> dict[str, dict[str, Any]]:
    """The part, by group, for the entries left out of each list in
    ``groups``: ``{group: {"counts": {...}, entries: [...]}}``, sent under
    ``field``, ``entries`` naming the key the named entries go under. ``keep`` leaves out the kinds
    a reply already names elsewhere. A group with no fact left out is absent,
    and so is the part when none has one. Never over ``bound`` estimated
    tokens when ``bound`` is :func:`worst_bound` for its vocabulary."""
    part: dict[str, dict[str, Any]] = {}
    named_kinds: dict[str, set[str]] = {}
    named = 0  # kinds named, over all groups
    for group, left in groups.items():
        counts: dict[str, int] = {}
        listed: list[dict[str, Any]] = []
        seen = named_kinds.setdefault(group, set())
        for index, identity, kinds in left.facts.entries:
            if index < left.start:
                continue
            kept = [kind for kind in kinds if keep(kind)]
            for kind in kept:
                counts[kind] = counts.get(kind, 0) + 1
            fresh = [kind for kind in kept if kind not in left.facts.every and kind not in seen]
            room = NAMED - named
            if fresh and room > 0:
                fresh = fresh[: min(room, KINDS_PER_ENTRY)]
                seen.update(fresh)
                named += len(fresh)
                entry: dict[str, Any] = {**identity, "kinds": fresh}
                cursor = left.cursor(index)
                if cursor is not None:
                    entry["cursor"] = cursor
                listed.append(entry)
        if counts:
            part[group] = {"counts": counts, **({entries: listed} if listed else {})}
    return _bounded(part, bound, entries, field)


def _bounded(
    part: dict[str, dict[str, Any]], bound: int, entries: str, field: str
) -> dict[str, Any]:
    """``part`` within ``bound``: named entries off from the last group, then
    the groups after the first summed, from the last, into :data:`FOLDED`."""

    def size() -> int:
        return estimate_tokens({field: part})

    for group in reversed(list(part)):
        while part[group].get(entries) and size() > bound:
            part[group][entries].pop()
            if not part[group][entries]:
                part[group].pop(entries)
    # The folded group's name, starred again past any group named so.
    into = FOLDED
    while into in part:
        into += FOLDED
    while size() > bound:
        groups = [group for group in part if group != into]
        if len(groups) <= 1:
            break
        folded = part.pop(groups[-1])
        summed = part.pop(into, {"counts": {}, "groups": 0})
        summed["groups"] += 1
        for kind, count in folded["counts"].items():
            summed["counts"][kind] = summed["counts"].get(kind, 0) + count
        part[into] = summed
    return part


def worst_bound(
    vocabulary: Sequence[str],
    *,
    identity: Mapping[str, Any],
    cursor: str | None,
    field: str,
    entries: str,
    group: str,
    folded: bool,
) -> int:
    """The size, in estimated tokens, of the largest part ``vocabulary``
    allows: every kind counted in the hundreds of thousands, :data:`NAMED`
    entries each named by ``identity`` (at its longest) for the longest kinds,
    with ``cursor``, in one group named ``group``, and, when ``folded``, a
    :data:`FOLDED` group counting every kind too."""
    longest = sorted(vocabulary, key=len, reverse=True)
    named = [
        {**identity, "kinds": longest[:KINDS_PER_ENTRY], **({"cursor": cursor} if cursor else {})}
        for _ in range(NAMED)
    ]
    part: dict[str, Any] = {group: {"counts": dict.fromkeys(vocabulary, 999_999), entries: named}}
    if folded:
        part[FOLDED] = {"counts": dict.fromkeys(vocabulary, 999_999), "groups": 999}
    return estimate_tokens({field: part})
