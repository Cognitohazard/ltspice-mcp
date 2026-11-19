"""How a complete answer is sent: whole when it fits, else stored and cut.

A tool computes its whole answer and hands it here with its output schema, a
target size and the identity of what the answer was computed from (its
snapshot). :func:`present` decides what is sent:

- An answer within the target is sent as it is: no field is added and
  nothing is stored.
- An answer over the target is stored whole, as one JSON file in the store's
  results folder (:meth:`Store.stored_answer`), under the expiry of
  ``[analysis] result_set_ttl_hours``. Its id is the digest of the answer and
  its snapshot, so the same answer of the same inputs is one file. Then the
  lists its schema declares ``items`` or ``rows`` (``lib/reply_cuts.py``) are
  cut from the end, whole entries only, the largest first, until the reply
  fits or each holds one entry. A reply that still does not fit goes out over
  the target. A list declared ``never``, a ``series`` or a list that declares
  nothing is never touched, and neither is anything inside the entries of any
  list. The reply then carries ``cut``, by each cut list's JSON Pointer, and
  ``stored_answer``.

A list the server already keeps elsewhere, such as the run rows a job's own
record holds, is a second kind of source (:class:`Record`): when it is cut, its
cursor names the record and an offset, nothing is stored for it, and a page is
read from the record when asked (:func:`open_record`, :func:`present_record_page`),
so it carries the record as it stands then. A reply whose cut lists are all
records writes no file.

A cursor names a stored answer, a list's pointer and an offset.
:func:`open_page` reads the stored file it names and :func:`present_page`
presents the entries of that list from the offset on, cut to the target the
same way. Nothing is computed again and no request is replayed, so a page is
never stale: it carries the stored answer's snapshot, which tells the caller
whether it predates a change to what was checked.

Only this module cuts a reply's lists for the tools that use it
(``tests/test_reply_cuts.py``). It holds no MCP types; ``tools/_base.py``
calls it beside ``reply_payload``.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

from ltspice_mcp.errors import InvalidCursorError
from ltspice_mcp.lib import now, parse_iso_datetime, read_text_file
from ltspice_mcp.lib.cursor_codec import CursorError, canonical_hash
from ltspice_mcp.lib.cursor_codec import decode_cursor as _decode_body
from ltspice_mcp.lib.cursor_codec import encode_cursor as _encode_body
from ltspice_mcp.lib.reply_cuts import ITEMS, schema_lists
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.store import KIND_STORED_ANSWER, Store, accept, atomic_write_json, envelope

logger = logging.getLogger(__name__)

#: The prefix of a stored answer's id; the rest is 32 hex digits.
STORED_PREFIX = "sa_"

#: The two kinds of list the presenter may cut.
_CUTTABLE = frozenset({ITEMS, "rows"})

_MISSING: Any = object()


@dataclass(frozen=True)
class ListCut:
    """What a reply holds of one list: ``returned`` entries from ``offset``,
    of ``total``, and the cursor (``next``) for the entries after them."""

    total: int
    returned: int
    offset: int = 0
    next: str | None = None

    def as_reply(self) -> dict[str, Any]:
        """The ``cut`` entry a reply carries, without the fields that say
        nothing: no ``offset`` on a first reply, no ``next`` on the last page."""
        entry: dict[str, Any] = {"total": self.total, "returned": self.returned}
        if self.offset:
            entry["offset"] = self.offset
        if self.next is not None:
            entry["next"] = self.next
        return entry


@dataclass(frozen=True)
class StoredAnswer:
    """A complete answer kept on disk, with what it was computed from."""

    id: str
    path: Path
    tool: str
    snapshot: dict[str, Any]
    answer: dict[str, Any] = field(repr=False)

    def as_reply(self) -> dict[str, Any]:
        """The ``stored_answer`` field a reply carries."""
        return {"id": self.id, "path": str(self.path), "snapshot": self.snapshot}


@dataclass(frozen=True)
class Record:
    """A list the server already keeps, named by the kind of record and its
    key (a job's run rows: the kind ``job_runs`` and the job's id): a cut of
    it is read back from the record, not from a stored copy."""

    kind: str
    key: str


@dataclass(frozen=True)
class Presented:
    """The reply to send, the lists it cut, the stored answer behind it, and
    its size against the target it was presented to."""

    reply: dict[str, Any]
    cut: dict[str, ListCut]
    stored: StoredAnswer | None
    size: int = 0
    target: int | None = None

    @property
    def met(self) -> bool:
        """Whether the reply is within its target (always, with no target)."""
        return self.target is None or self.size <= self.target


#: Adds the tool's own words about a cut to a reply: ``(reply, cut, stored,
#: paged)``, where ``stored`` is None when every cut list is a record, and
#: ``paged`` is the pointer of the list a page reads, or None on a first
#: reply. Called on each candidate, so what it adds is measured.
Annotate = Callable[[dict[str, Any], Mapping[str, ListCut], StoredAnswer | None, str | None], None]

#: Says, once, that a reply went out over its target: ``(reply, size, target)``.
#: Called on the reply sent, after the cut, only when it is still over.
Over = Callable[[dict[str, Any], int, int], None]


# ---------------------------------------------------------------------------
# JSON Pointers
# ---------------------------------------------------------------------------


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _segments(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"not a JSON Pointer: {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]


def split_pointer(pointer: str) -> list[str]:
    """The keys a JSON Pointer names, unescaped."""
    return _segments(pointer)


def join_pointer(keys: Iterable[str]) -> str:
    """The JSON Pointer of ``keys``, escaped."""
    return "".join(f"/{_escape(str(key))}" for key in keys)


def resolve_pointer(document: Any, pointer: str, *, default: Any = _MISSING) -> Any:
    """The value at ``pointer`` in ``document``; ``default`` when absent, else
    ``KeyError``."""
    node = document
    for part in _segments(pointer):
        try:
            node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, IndexError, ValueError, TypeError):
            if default is _MISSING:
                raise KeyError(pointer) from None
            return default
    return node


# ---------------------------------------------------------------------------
# Which lists of an answer may be cut
# ---------------------------------------------------------------------------

_BRANCH = re.compile(r"<(?:anyOf|oneOf|allOf)\d+>")


def _unbranched(path: str) -> str:
    """A schema path without its union branches, as the payload walk spells it."""
    plain = _BRANCH.sub("", path)
    if plain.startswith(".") and not plain.startswith(".*"):
        plain = plain[1:]
    return plain


def _declarations(schema: Mapping[str, Any]) -> tuple[dict[str, str | None], set[str]]:
    """Each declared list's path with its kind, and every prefix of those
    paths. A path declared differently in two branches of a union is cut only
    when every branch lets it be."""
    kinds: dict[str, str | None] = {}
    for item in schema_lists(schema):
        path = _unbranched(item.path)
        kind = item.kind
        if path in kinds and kinds[path] != kind:
            kind = kind if kinds[path] in _CUTTABLE and kind in _CUTTABLE else None
        kinds[path] = kind
    prefixes: set[str] = set()
    for path in kinds:
        for match in re.finditer(r"\.|\[\]", path):
            prefixes.add(path[: match.start()])
        prefixes.add(path)
    return kinds, prefixes


def cuttable_lists(answer: Mapping[str, Any], schema: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The lists of ``answer`` the presenter may cut, with each one's JSON
    Pointer and declared kind, in the order they appear.

    The walk enters objects and the values of maps, never a list's entries,
    so a list inside a row is cut only with the whole row.
    """
    kinds, prefixes = _declarations(schema)
    found: list[tuple[str, str]] = []

    def walk(node: Any, path: str, pointer: str) -> None:
        if isinstance(node, list):
            kind = kinds.get(path)
            if kind in _CUTTABLE:
                assert kind is not None
                found.append((pointer, kind))
            return
        if not isinstance(node, dict):
            return
        for key, child in node.items():
            named = f"{path}.{key}" if path else str(key)
            mapped = f"{path}.*"
            if named in prefixes:
                walk(child, named, f"{pointer}/{_escape(str(key))}")
            elif mapped in prefixes:
                walk(child, mapped, f"{pointer}/{_escape(str(key))}")

    walk(answer, "", "")
    return found


# ---------------------------------------------------------------------------
# Building and measuring a reply
# ---------------------------------------------------------------------------


def _with_lists(answer: dict[str, Any], lists: Mapping[str, list[Any] | None]) -> dict[str, Any]:
    """``answer`` with the list at each pointer replaced, or removed for None,
    copying only the objects on the way to each."""
    reply = dict(answer)
    for pointer, value in lists.items():
        parts = _segments(pointer)
        node = reply
        for part in parts[:-1]:
            copied = dict(node[part])
            node[part] = copied
            node = copied
        if value is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    return reply


def _without(reply: dict[str, Any], pointers: Iterable[str]) -> dict[str, Any]:
    """``reply`` without the fields at ``pointers`` that it holds."""
    absent = object()
    present = {
        pointer: None
        for pointer in pointers
        if resolve_pointer(reply, pointer, default=absent) is not absent
    }
    return _with_lists(reply, present) if present else reply


def _size(reply: Mapping[str, Any]) -> int:
    return estimate_tokens(reply, as_sent=False)


def _fit(
    sources: Mapping[str, list[Any]],
    build: Callable[[Mapping[str, int]], dict[str, Any]],
    target: int,
    caps: Mapping[str, int] | None = None,
    apart: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """How many entries of each list in ``sources`` the reply keeps.

    A list starts at its cap, when ``caps`` names one (a count the caller set,
    such as a page size). The largest list still holding more than one entry
    is cut first, to the most entries that let the reply fit; when none does,
    to one entry, and the next largest is cut. A list whose cut to one entry
    would make the reply no shorter (a short list, whose ``cut`` entry costs
    more than its entries) is left as it is.

    ``apart`` names, by JSON Pointer, the fields of a reply that are sized
    apart, each with the bound it reserves: a note the annotation adds on the
    entries left out, which shrinks as more are kept. The entries are sized
    against the target less those bounds, without those fields, so the size
    grows with every entry kept and a binary search is right. The reserve is
    a first estimate: the whole reply at the count found decides, and while it
    is over, one entry comes off; a list that is over at one entry is held
    there and the search goes on to the next. Where the search finds no
    count, every count that fits without the reserve is tried whole, from
    the largest. A reply is over its target only when the lists, each at the
    least this search would cut it to, are over it together.
    """
    apart = apart or {}
    reserve = sum(apart.values())

    def bare(keep: Mapping[str, int]) -> int:
        return _size(_without(build(keep), apart))

    def whole(keep: Mapping[str, int]) -> int:
        return _size(build(keep))

    keep = {
        pointer: min(len(entries), max(1, (caps or {}).get(pointer, len(entries))))
        for pointer, entries in sources.items()
    }
    current = whole(keep)
    if current <= target:
        return keep
    settled: set[str] = set()
    while True:
        open_lists = [
            pointer for pointer, count in keep.items() if count > 1 and pointer not in settled
        ]
        if not open_lists:
            return keep
        largest = max(
            open_lists,
            key=lambda p: len(json.dumps(sources[p][: keep[p]], default=str)),
        )
        settled.add(largest)
        best = _largest(
            lambda count, at=largest: bare({**keep, at: count}) + reserve <= target,
            keep[largest] - 1,
        )
        if best is None and apart:
            most = _largest(
                lambda count, at=largest: bare({**keep, at: count}) <= target, keep[largest] - 1
            )
            # The note shrinks as entries are kept, so the counts that fit
            # whole can be a band away from both ends: every count is tried,
            # from the largest. This runs only when no count fits beside the
            # reserve, a target near the reserve, so the counts are few.
            best = next(
                (
                    count
                    for count in range(most or 0, 0, -1)
                    if whole({**keep, largest: count}) <= target
                ),
                None,
            )
        # The reserve is a first estimate: the note at the count chosen can
        # be larger than at one entry (the entries left out change). The
        # whole reply at that count decides; while it is over, one entry
        # comes off.
        while best is not None and best > 1 and whole({**keep, largest: best}) > target:
            best -= 1
        if best is not None:
            size = whole({**keep, largest: best})
            if size <= target:
                keep[largest] = best
                return keep
        # No count of this list fits with the others as they are: it is held
        # at one entry, when that shortens the reply, and the next is cut.
        size = whole({**keep, largest: 1})
        if size < current:
            keep[largest], current = 1, size


def _largest(fits: Callable[[int], bool], most: int) -> int | None:
    """The largest count from 1 to ``most`` that fits, for a ``fits`` that
    holds up to some count and fails above it; None when 1 does not."""
    return _largest_from(fits, 1, most)


def _largest_from(fits: Callable[[int], bool], least: int, most: int) -> int | None:
    """The largest count from ``least`` to ``most`` that fits, as
    :func:`_largest`; None when ``least`` does not."""
    low, high, best = least, most, None
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


# ---------------------------------------------------------------------------
# The stored answer
# ---------------------------------------------------------------------------


def _answer_id(tool: str, answer: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    digest = canonical_hash({"tool": tool, "snapshot": snapshot, "answer": answer})
    return STORED_PREFIX + digest[:32]


def is_stored_answer_id(text: object) -> bool:
    """Whether ``text`` has the form of a stored answer's id."""
    return (
        isinstance(text, str)
        and text.startswith(STORED_PREFIX)
        and len(text) == len(STORED_PREFIX) + 32
        and all(char in "0123456789abcdef" for char in text[len(STORED_PREFIX) :])
    )


# Not result_store's expiry: a result set computed from jobs lives as long as
# its jobs, while a stored answer depends on no job and expires by time alone.
# Both read [analysis] result_set_ttl_hours; result_store.cleanup says the same.
def _expired(data: Mapping[str, Any]) -> bool:
    expires = parse_iso_datetime(str(data.get("expires_at", "")))
    return expires is None or expires <= now()


#: The least time, in seconds, between two sweeps a cut reply runs.
SWEEP_INTERVAL_S: Final = 600.0

def cleanup(store: Store) -> int:
    """Delete the stored answers past their expiry; how many were deleted.

    A stored answer's file has its expiry as its modification time
    (:func:`_store_answer`), so the sweep reads no answer: one stat per file.
    """
    removed = 0
    try:
        paths = list(store.results_dir.glob(f"{STORED_PREFIX}*.json"))
    except OSError:
        return 0
    moment = now().timestamp()
    for path in paths:
        try:
            if path.stat().st_mtime <= moment:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _sweep_if_due(store: Store) -> None:
    """Sweep when the last sweep is :data:`SWEEP_INTERVAL_S` old or more: a
    cut reply spends one stat here, and on a due sweep one more per stored
    answer, reading none."""
    marker = store.stored_answers_swept()
    moment = now().timestamp()
    try:
        last = marker.stat().st_mtime
    except OSError:
        last = None
    if last is not None and moment - last < SWEEP_INTERVAL_S:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        os.utime(marker, (moment, moment))
    except OSError:
        return
    cleanup(store)


def _store_answer(store: Store, stored: StoredAnswer, ttl_hours: float) -> None:
    """Write ``stored`` to its file, with its expiry as the file's
    modification time, after a sweep when one is due."""
    _sweep_if_due(store)
    created = now()
    expires = created + timedelta(hours=ttl_hours)
    store.ensure_root()
    stored.path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        stored.path,
        envelope(
            KIND_STORED_ANSWER,
            id=stored.id,
            tool=stored.tool,
            created_at=created.isoformat(),
            expires_at=expires.isoformat(),
            snapshot=stored.snapshot,
            answer=stored.answer,
        ),
    )
    stamp = expires.timestamp()
    os.utime(stored.path, (stamp, stamp))


#: The code of the note a cut reply carries when its answer was not kept.
NOT_KEPT_CODE: Final = "stored_answer_not_kept"


def not_kept_note(reason: str) -> dict[str, str]:
    """What a cut reply says when its complete answer could not be written:
    it cannot be read on, why, and what to ask instead."""
    return {
        "code": NOT_KEPT_CODE,
        "message": (
            f"The complete answer could not be kept ({reason}), so this cut reply "
            "cannot be read on and its cut entries carry no cursor. Ask again with a "
            "larger budget or a narrower request; analyze_results can also write a "
            "waveform's every sample to a file with format 'csv'."
        ),
    }


def _cursor(answer_id: str, pointer: str, offset: int) -> str:
    return _encode_body({"answer": answer_id, "list": pointer, "offset": offset})


def stored_cursor(stored: StoredAnswer, pointer: str, offset: int) -> str:
    """The cursor that reads the list at ``pointer`` of ``stored`` from ``offset``."""
    return _cursor(stored.id, pointer, offset)


def record_cursor(record: Record, offset: int) -> str:
    """The cursor that reads ``record``'s list from ``offset``."""
    return _encode_body({"record": record.kind, "key": record.key, "offset": offset})


# ---------------------------------------------------------------------------
# Presenting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One answer of a reply, with what presenting it needs: where it sits in
    the reply (``root``, a JSON Pointer; ``""`` for the reply itself), the
    output schema of its tool, the tool it is an answer of, and what it was
    computed from (``snapshot``, or a function called only when it is
    stored). ``annotate``, ``records``, ``caps`` and ``apart`` are as for
    :func:`present`, each named relative to ``root``."""

    root: str
    schema: Mapping[str, Any]
    tool: str
    snapshot: dict[str, Any] | Callable[[], dict[str, Any]]
    annotate: Annotate | None = None
    records: Mapping[str, Record] | None = None
    caps: Mapping[str, int] | None = None
    apart: Mapping[str, int] | None = None


@dataclass(frozen=True)
class Series:
    """How a tool redraws the samples a schema declares as one ``series``:
    ``redraw(holder, points)`` gives the object that holds them at no more
    than ``points`` points, aligned (a waveform's axis and its values). A
    thinned series is never paged: its ``cut`` entry gives the points held
    and returned, with no cursor. Never thinned below ``floor`` points."""

    redraw: Callable[[dict[str, Any], int], dict[str, Any]]
    floor: int = 32


def _within(pointer: str, root: str) -> bool:
    return root == "" or pointer == root or pointer.startswith(root + "/")


def _relative(pointer: str, root: str) -> str:
    return pointer[len(root) :] if root else pointer


@dataclass
class _Placed:
    """One answer while a reply is presented: its global pointers."""

    answer: Answer
    stored: StoredAnswer | None = None
    #: Why the stored copy could not be written, when it could not.
    not_kept: str | None = None
    records: dict[str, Record] = field(default_factory=dict)
    caps: dict[str, int] = field(default_factory=dict)
    apart: dict[str, int] = field(default_factory=dict)


def _place(answers: Sequence[Answer]) -> list[_Placed]:
    placed = []
    for answer in answers:
        root = answer.root

        def at(relative: str, root: str = root) -> str:
            # A pointer relative to the answer's root, or a field at its root.
            return root + (relative if relative.startswith("/") else "/" + _escape(relative))

        placed.append(
            _Placed(
                answer,
                records={at(p): r for p, r in (answer.records or {}).items()},
                caps={at(p): c for p, c in (answer.caps or {}).items()},
                apart={at(p): n for p, n in (answer.apart or {}).items()},
            )
        )
    return placed


def _owner(pointer: str, placed: Sequence[_Placed]) -> _Placed:
    """The answer a global pointer belongs to: the deepest whose root holds it."""
    return max(
        (part for part in placed if _within(pointer, part.answer.root)),
        key=lambda part: len(part.answer.root),
    )


def _all_lists(document: Any, placed: Sequence[_Placed]) -> list[tuple[str, str]]:
    """Every cuttable list of every answer, by global pointer, each found by
    its own answer's schema and owned by the deepest answer that holds it."""
    found: list[tuple[str, str]] = []
    for part in placed:
        root = part.answer.root
        node = resolve_pointer(document, root, default=None)
        if not isinstance(node, dict):
            continue
        for pointer, kind in cuttable_lists(node, part.answer.schema):
            whole = root + pointer
            if _owner(whole, placed) is part:
                found.append((whole, kind))
    return found


def _series_lists(document: Any, schema: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each list ``schema`` declares a ``series``, by the pointer of the
    object holding it and the list's key, entering the entries of lists."""
    kinds: dict[str, str | None] = {}
    for item in schema_lists(schema):
        kinds[_unbranched(item.path)] = item.kind
    prefixes: set[str] = set()
    for path in kinds:
        for match in re.finditer(r"\.|\[\]", path):
            prefixes.add(path[: match.start()])
        prefixes.add(path)
    found: list[tuple[str, str]] = []

    def walk(node: Any, path: str, pointer: str) -> None:
        if isinstance(node, list):
            entry = f"{path}[]"
            if entry in prefixes:
                for index, child in enumerate(node):
                    walk(child, entry, f"{pointer}/{index}")
            return
        if not isinstance(node, dict):
            return
        for key, child in node.items():
            named = f"{path}.{key}" if path else str(key)
            mapped = f"{path}.*"
            chosen = named if named in prefixes else mapped if mapped in prefixes else None
            if chosen is None:
                continue
            if isinstance(child, list) and kinds.get(chosen) == "series":
                found.append((pointer, str(key)))
                continue
            walk(child, chosen, f"{pointer}/{_escape(str(key))}")

    walk(document, "", "")
    return found


def _thin(
    document: dict[str, Any],
    placed: Sequence[_Placed],
    series: Series | None,
    target: int | None,
    measure: Callable[[dict[str, Any], Mapping[str, ListCut]], int],
) -> tuple[dict[str, Any], dict[str, ListCut]]:
    """``document`` with every series redrawn at the most points that let the
    reply fit ``target``, never below the floor, and the ``cut`` entry of each
    list thinned. ``measure`` sizes the reply presented from a document with
    those entries, every list at its cap, so the words a cut adds are part of
    what fits. Unchanged when it fits, or nothing is declared a series."""
    if series is None or target is None or measure(document, {}) <= target:
        return document, {}
    holders: dict[str, list[str]] = {}
    for part in placed:
        root = part.answer.root
        node = resolve_pointer(document, root, default=None)
        if not isinstance(node, dict):
            continue
        for holder, key in _series_lists(node, part.answer.schema):
            holders.setdefault(root + holder, []).append(key)
    if not holders:
        return document, {}
    originals = {pointer: resolve_pointer(document, pointer) for pointer in holders}
    held = {
        f"{pointer}/{_escape(key)}": len(originals[pointer][key])
        for pointer, keys in holders.items()
        for key in keys
    }
    most = max(held.values(), default=0)
    if most <= series.floor:
        return document, {}

    @functools.cache
    def at(points: int) -> tuple[dict[str, Any], dict[str, ListCut]]:
        drawn = _with_values(
            document,
            {pointer: series.redraw(holder, points) for pointer, holder in originals.items()},
        )
        cuts: dict[str, ListCut] = {}
        for pointer, total in held.items():
            shown = len(resolve_pointer(drawn, pointer))
            if shown < total:
                cuts[pointer] = ListCut(total=total, returned=shown)
        return drawn, cuts

    def kept(count: int) -> bool:
        drawn, _ = at(count)
        return all(
            len(resolve_pointer(drawn, pointer)) >= min(series.floor, total)
            for pointer, total in held.items()
        )

    # A drawing can hold fewer points than it is asked for (a bucket with no
    # sample), so the floor is held on what is drawn: the fewest points asked
    # for whose drawing keeps it, then the most that fit from there.
    low, high = series.floor, most
    while low < high:
        middle = (low + high) // 2
        if kept(middle):
            high = middle
        else:
            low = middle + 1
    if low >= most:
        return document, {}
    points = _largest_from(lambda count: measure(*at(count)) <= target, low, most - 1)
    return at(points if points is not None else low)


def _with_values(answer: dict[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """``answer`` with the value at each pointer replaced, copying the
    containers on the way to each (lists included)."""
    reply = dict(answer)
    for pointer, value in values.items():
        parts = _segments(pointer)
        node: Any = reply
        for part in parts[:-1]:
            child = node[int(part)] if isinstance(node, list) else node[part]
            copied = list(child) if isinstance(child, list) else dict(child)
            if isinstance(node, list):
                node[int(part)] = copied
            else:
                node[part] = copied
            node = copied
        if isinstance(node, list):
            node[int(parts[-1])] = value
        else:
            node[parts[-1]] = value
    return reply


def _present_lists(
    base: dict[str, Any],
    sources: Mapping[str, list[Any]],
    offsets: Mapping[str, int],
    totals: Mapping[str, int],
    placed: Sequence[_Placed],
    *,
    target: int,
    paged: str | None,
    fixed: Mapping[str, ListCut] | None = None,
    over: Over | None = None,
    counts: Mapping[str, int] | None = None,
) -> Presented:
    """``base`` with each list of ``sources`` cut to fit ``target``. A list
    is named in ``cut`` of the answer that owns it when the reply holds less
    than all of it, or when it is the list a page reads (``paged``);
    ``fixed`` names lists the reply leaves out whole or thinned. A list in
    its answer's ``records`` continues by a record cursor; every other by a
    cursor into its answer's stored copy. Every pointer here is global: each
    answer's ``cut`` names its lists relative to its root. ``counts``, when
    given, is the count of each list the reply keeps, decided already."""
    owners = {pointer: _owner(pointer, placed) for pointer in [*sources, *(fixed or {})]}
    caps = {pointer: cap for part in placed for pointer, cap in part.caps.items()}
    apart = {pointer: bound for part in placed for pointer, bound in part.apart.items()}

    def next_cursor(pointer: str, end: int) -> str | None:
        part = owners[pointer]
        if pointer in part.records:
            return record_cursor(part.records[pointer], end)
        if part.not_kept is not None:
            # No copy to read on from: the entry says how many it holds, and
            # the answer's stored_answer says why there is no cursor.
            return None
        assert part.stored is not None, f"{pointer} is cut and is neither stored nor a record"
        return _cursor(part.stored.id, _relative(pointer, part.answer.root), end)

    def cuts(keep: Mapping[str, int]) -> dict[str, ListCut]:
        made: dict[str, ListCut] = {}
        for pointer in sources:
            offset, total = offsets.get(pointer, 0), totals[pointer]
            end = offset + keep[pointer]
            if end >= total and offset == 0 and pointer != paged:
                continue
            made[pointer] = ListCut(
                total=total,
                returned=keep[pointer],
                offset=offset,
                next=next_cursor(pointer, end) if end < total else None,
            )
        made.update(fixed or {})
        return made

    def build(keep: Mapping[str, int]) -> dict[str, Any]:
        reply = _with_lists(
            base,
            {pointer: entries[: keep[pointer]] or None for pointer, entries in sources.items()},
        )
        made = cuts(keep)
        # Outermost first, so an inner answer's fields are written into the
        # object its outer answer's root copy holds.
        for part in sorted(placed, key=lambda part: len(part.answer.root)):
            root = part.answer.root
            node = resolve_pointer(reply, root, default=None)
            if not isinstance(node, dict):
                continue
            if root:
                node = dict(node)
                reply = _with_values(reply, {root: node})
            mine = {
                _relative(pointer, root): entry
                for pointer, entry in made.items()
                if owners.get(pointer, _owner(pointer, placed)) is part
            }
            node["cut"] = {pointer: entry.as_reply() for pointer, entry in mine.items()}
            if part.stored is not None:
                node["stored_answer"] = part.stored.as_reply()
            elif part.not_kept is not None:
                node["stored_answer"] = {"not_kept": not_kept_note(part.not_kept)}
            if part.answer.annotate is not None:
                part.answer.annotate(
                    node,
                    mine,
                    part.stored,
                    _relative(paged, root) if paged is not None and _within(paged, root) else None,
                )
        return reply

    keep = dict(counts) if counts is not None else _fit(sources, build, target, caps, apart)
    reply = build(keep)
    size = _size(reply)
    if size > target and over is not None:
        over(reply, size, target)
    return Presented(reply, cuts(keep), placed[0].stored if placed else None, size, target)


def present(
    answer: dict[str, Any],
    schema: Mapping[str, Any],
    *,
    target: int | None,
    tool: str,
    snapshot: dict[str, Any] | Callable[[], dict[str, Any]],
    store: Store,
    ttl_hours: float,
    annotate: Annotate | None = None,
    records: Mapping[str, Record] | None = None,
    caps: Mapping[str, int] | None = None,
    over: Over | None = None,
    apart: Mapping[str, int] | None = None,
    series: Series | None = None,
) -> Presented:
    """Send ``answer`` whole when it fits ``target`` tokens, or cut its
    cuttable lists to fit, storing it when a list that is not a record is cut.

    ``answer`` is the reply as ``reply_payload`` makes it. A ``target`` of
    None sends every answer whole. ``snapshot`` may be a function, called
    only when the answer is stored, so an answer that fits reads nothing.
    ``records`` names, by pointer, the lists the server already keeps, which
    continue from their record and are never stored. ``caps`` names, by
    pointer, the most entries a list may send, whatever the target. ``over``
    says, once, that the reply sent is still over the target. ``apart`` names
    the fields an annotation adds that are sized apart, with their bounds
    (:func:`_fit`). ``series`` redraws the samples the schema declares a
    series, first, at the most points that fit (:func:`_thin`).
    """
    return present_many(
        answer,
        [Answer("", schema, tool, snapshot, annotate, records, caps, apart)],
        target=target,
        store=store,
        ttl_hours=ttl_hours,
        over=over,
        series=series,
    )


def present_many(
    reply: dict[str, Any],
    answers: Sequence[Answer],
    *,
    target: int | None,
    store: Store,
    ttl_hours: float,
    over: Over | None = None,
    series: Series | None = None,
) -> Presented:
    """Present several answers under one target: ``reply`` holds each of
    ``answers`` at its root (the first at ``""``, the reply itself). One size
    search runs over the lists of every answer at once, so a reply that fits
    goes out whole and no answer is cut while room remains. Each answer keeps
    its own tool, snapshot and records, and carries its own ``cut`` and
    ``stored_answer`` at its root: a cut list of an answer continues from
    that answer's stored copy, under its own tool, or from its record.
    """
    placed = _place(answers)
    caps = {pointer: cap for part in placed for pointer, cap in part.caps.items()}
    capped = any(
        len(entries) > cap
        for pointer, cap in caps.items()
        if isinstance(entries := resolve_pointer(reply, pointer, default=None), list)
    )
    size = _size(reply) if target is not None else 0
    if (target is None or size <= target) and not capped:
        return Presented(reply, {}, None, size, target)
    lists = _all_lists(reply, placed)
    pointers = [pointer for pointer, _ in lists if len(resolve_pointer(reply, pointer)) > 1]
    # Stored as computed, before any series is thinned: a page thins its own.
    for part in placed:
        if any(_owner(p, placed) is part and p not in part.records for p in pointers):
            node = resolve_pointer(reply, part.answer.root)
            taken = (
                part.answer.snapshot() if callable(part.answer.snapshot) else part.answer.snapshot
            )
            answer_id = _answer_id(part.answer.tool, node, taken)
            part.stored = StoredAnswer(
                answer_id, store.stored_answer(answer_id), part.answer.tool, taken, node
            )
    full_target = sys.maxsize if target is None else target

    totals = {pointer: len(resolve_pointer(reply, pointer)) for pointer in pointers}

    def presented(
        document: dict[str, Any],
        thinned: Mapping[str, ListCut],
        counts: Mapping[str, int] | None = None,
    ) -> Presented:
        sources = {pointer: resolve_pointer(document, pointer) for pointer in pointers}
        return _present_lists(
            document,
            sources,
            {},
            totals,
            placed,
            target=full_target,
            paged=None,
            fixed=thinned,
            over=over,
            counts=counts,
        )

    def measured(document: dict[str, Any], thinned: Mapping[str, ListCut]) -> int:
        sources = {pointer: resolve_pointer(document, pointer) for pointer in pointers}
        return _present_lists(
            document,
            sources,
            {},
            totals,
            placed,
            target=sys.maxsize,
            paged=None,
            fixed=thinned,
        ).size

    document, thinned = _thin(reply, placed, series, target, measured)
    if not pointers and not thinned:
        if target is not None and size > target and over is not None:
            document = dict(document)
            over(document, size, target)
        return Presented(document, {}, None, size, target)
    shown = presented(document, thinned)
    if thinned:
        # The series were thinned for every entry; draw them again for the
        # entries the reply keeps, so an entry left out neither thins those
        # shown nor is named in cut.
        kept = {
            pointer: entry.returned for pointer, entry in shown.cut.items() if pointer in totals
        }
        if kept:
            sliced = _with_lists(
                reply, {p: resolve_pointer(reply, p)[:count] or None for p, count in kept.items()}
            )
            document, thinned = _thin(sliced, placed, series, target, measured)
            shown = presented(document, thinned)
    redo = False
    for part in placed:
        if part.stored is None:
            continue
        if any(
            entry.next is not None
            for pointer, entry in shown.cut.items()
            if _owner(pointer, placed) is part and pointer not in part.records
        ):
            try:
                _store_answer(store, part.stored, ttl_hours)
            except OSError as exc:
                logger.warning("stored answer %s not kept: %s", part.stored.id, exc)
                part.stored, part.not_kept = None, f"{type(exc).__name__}: {exc}"
                redo = True
        else:
            # Every list of this answer that is not a record went out whole.
            part.stored = None
            redo = True
    if redo:
        # The same counts, without the stored copy no list continues from:
        # a search again could cut a list the reply now has no copy of.
        counts = {
            pointer: shown.cut[pointer].returned if pointer in shown.cut else len(entries)
            for pointer in pointers
            if isinstance(entries := resolve_pointer(document, pointer), list)
        }
        shown = presented(document, thinned, counts)
    return replace(shown, target=target)


def present_page(
    stored: StoredAnswer,
    pointer: str,
    offset: int,
    schema: Mapping[str, Any],
    *,
    target: int | None,
    annotate: Annotate | None = None,
    cap: int | None = None,
    over: Over | None = None,
    apart: Mapping[str, int] | None = None,
    series: Series | None = None,
) -> Presented:
    """The entries of the list at ``pointer`` from ``offset`` on, read from
    ``stored`` and cut to fit ``target``, and to ``cap`` entries when given.

    A page holds its list's entries and what identifies it, and nothing the
    first reply already gave (:func:`page_skeleton`): ``page_of`` names the
    list, ``cut`` names only it (where it starts, how many it holds, the
    cursor for the rest) and ``stored_answer`` the answer it reads. Its
    series are thinned as a first reply's are (``series``).
    """
    answer = stored.answer
    whole = resolve_pointer(answer, pointer)
    placed = _place(
        [Answer("", schema, stored.tool, stored.snapshot, annotate, None, None, apart)]
    )
    placed[0].stored = stored
    if cap is not None:
        placed[0].caps[pointer] = cap
    base = {
        **_with_lists(page_skeleton(answer, schema, pointer), {pointer: whole[offset:]}),
        PAGE_OF: pointer,
    }
    left_out: dict[str, ListCut] = {}

    def page(
        document: dict[str, Any], thinned: Mapping[str, ListCut], aim: int, said: Over | None
    ) -> Presented:
        return _present_lists(
            document,
            {pointer: resolve_pointer(document, pointer)},
            {pointer: offset},
            {pointer: len(whole)},
            placed,
            target=aim,
            paged=pointer,
            fixed={**thinned, **left_out},
            over=said,
        )

    aim = sys.maxsize if target is None else target

    def measured(doc: dict[str, Any], cuts: Mapping[str, ListCut]) -> int:
        return page(doc, cuts, sys.maxsize, None).size

    document, thinned = _thin(base, placed, series, target, measured)
    shown = page(document, thinned, aim, over)
    entry = shown.cut.get(pointer)
    if thinned and entry is not None and entry.returned < len(whole) - offset:
        # Drawn again for the entries the page keeps (as present_many does).
        sliced = _with_lists(base, {pointer: whole[offset : offset + entry.returned]})
        document, thinned = _thin(sliced, placed, series, target, measured)
        shown = page(document, thinned, aim, over)
    return shown


#: The field a page carries: the JSON Pointer of the list it is a page of, so
#: a field it leaves out is read as given by the first reply, not as absent.
PAGE_OF: Final = "page_of"


def page_skeleton(answer: Mapping[str, Any], schema: Mapping[str, Any], pointer: str) -> dict:
    """What a page of the list at ``pointer`` keeps of ``answer``: the objects
    on the way to the list, each with its text, number and flag fields and
    the objects its schema requires in their own least form, and the list. Everything else the first reply gave, and the page leaves out."""
    keys = split_pointer(pointer)

    def child_schema(node_schema: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        properties = node_schema.get("properties") or {}
        if key in properties:
            return properties[key]
        extra = node_schema.get("additionalProperties")
        return extra if isinstance(extra, Mapping) else {}

    def least(node: Mapping[str, Any], node_schema: Mapping[str, Any]) -> dict[str, Any]:
        # Its text, number and flag fields as they are (a tool's action, a
        # recipe's metric, a page's counts), each object its schema requires
        # in its own least form; every list and every other object is left
        # out.
        required = set(node_schema.get("required") or ())
        kept: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, str | int | float):
                kept[key] = value
            elif isinstance(value, Mapping) and key in required:
                kept[key] = least(value, child_schema(node_schema, key))
        return kept

    def keep(node: Any, node_schema: Mapping[str, Any], rest: list[str]) -> Any:
        if not rest or not isinstance(node, Mapping):
            return node
        kept = least({k: v for k, v in node.items() if k != rest[0]}, node_schema)
        if rest[0] in node:
            kept[rest[0]] = keep(node[rest[0]], child_schema(node_schema, rest[0]), rest[1:])
        return kept

    return keep(answer, schema, keys)


def present_record_page(
    answer: dict[str, Any],
    pointer: str,
    offset: int,
    record: Record,
    *,
    target: int | None,
    annotate: Annotate | None = None,
    over: Over | None = None,
    cap: int | None = None,
    apart: Mapping[str, int] | None = None,
) -> Presented:
    """The entries of the record's list at ``pointer`` from ``offset`` on, cut
    to fit ``target``, and to ``cap`` entries when given; ``apart`` as for
    :func:`present`.

    ``answer`` is the complete answer read from the record now, as the tool
    makes it, so a page carries the record as it stands when it is read.
    Nothing is stored. Every other field of ``answer`` is sent as it is. Read
    from the start, an answer that fits is sent whole, as :func:`present`
    sends it.
    """
    whole = resolve_pointer(answer, pointer)
    if offset == 0 and (cap is None or len(whole) <= cap):
        size = _size(answer) if target is not None else 0
        if target is None or size <= target:
            return Presented(answer, {}, None, size, target)
    placed = _place([Answer("", {}, "", {}, annotate, {pointer: record}, None, apart)])
    if cap is not None:
        placed[0].caps[pointer] = cap
    return _present_lists(
        answer,
        {pointer: whole[offset:]},
        {pointer: offset},
        {pointer: len(whole)},
        placed,
        target=sys.maxsize if target is None else target,
        paged=pointer,
        over=over,
    )


def open_record(text: str, *, kind: str) -> tuple[str, int]:
    """The key and offset a record cursor of ``kind`` names.

    ``InvalidCursorError`` for anything else: a cursor this module did not
    write, a stored answer's cursor, or a cursor for another kind of record.
    Whether the offset still names an entry is the reader's to check, since
    only it reads the record.
    """
    refused = _refusal(text, "this action")
    try:
        body = _decode_body(text)
    except CursorError:
        raise InvalidCursorError(refused) from None
    key, offset = body.get("key"), body.get("offset")
    if not (
        body.get("record") == kind
        and isinstance(key, str)
        and key
        and type(offset) is int
        and offset >= 0
    ):
        raise InvalidCursorError(refused)
    return key, offset


def _refusal(text: str, returned_by: str) -> str:
    """Why ``text`` is refused as a cursor. An offset cursor of an earlier
    release (``o:<offset>``) is named as one, with the way to start again."""
    said = (
        f"{text[:40]!r} is not a cursor {returned_by} returned: pass a cut entry's next unchanged."
    )
    if text.startswith("o:"):
        said += (
            " Offset cursors from earlier releases are not read; the same call with no "
            "cursor starts again from the first entry."
        )
    return said


def open_page(
    text: str, schema: Mapping[str, Any], *, tool: str, store: Store, gone: str
) -> tuple[StoredAnswer, str, int]:
    """The stored answer a cursor reads, with its list's pointer and offset.

    ``InvalidCursorError`` for a cursor this module did not write, one for
    another tool's answer, one naming a list ``schema`` does not let be cut
    (the codec's checksum is no secret, so any body decodes), and one whose
    stored answer has expired or is missing, which says so and adds
    ``gone``: how to get a fresh answer.
    """
    refused = _refusal(text, "this tool")
    try:
        body = _decode_body(text)
    except CursorError:
        raise InvalidCursorError(refused) from None
    answer_id, pointer, offset = body.get("answer"), body.get("list"), body.get("offset")
    if not (
        is_stored_answer_id(answer_id)
        and isinstance(pointer, str)
        and (pointer == "" or pointer.startswith("/"))
        and type(offset) is int
        and offset >= 0
    ):
        raise InvalidCursorError(refused)
    assert isinstance(answer_id, str)
    path = store.stored_answer(answer_id)
    missing = f"The stored answer {answer_id} is gone: it expired or was removed. {gone}"
    try:
        data = json.loads(read_text_file(path))
    except FileNotFoundError:
        raise InvalidCursorError(missing, show_hint=False) from None
    except (OSError, ValueError) as exc:
        raise InvalidCursorError(
            f"The stored answer {answer_id} cannot be read: {exc}. {gone}"
        ) from None
    if not accept(data, path, kind=KIND_STORED_ANSWER, log=logger):
        raise InvalidCursorError(f"The stored answer {answer_id} is not readable here. {gone}")
    if _expired(data):
        with contextlib.suppress(OSError):
            path.unlink()
        raise InvalidCursorError(missing, show_hint=False)
    answer, snapshot = data.get("answer"), data.get("snapshot")
    if data.get("tool") != tool or not isinstance(answer, dict):
        raise InvalidCursorError(refused)
    stored = StoredAnswer(
        answer_id, path, tool, dict(snapshot) if isinstance(snapshot, dict) else {}, answer
    )
    whole = resolve_pointer(answer, pointer, default=None)
    if (
        pointer not in {listed for listed, _ in cuttable_lists(answer, schema)}
        or not isinstance(whole, list)
        or offset >= len(whole)
    ):
        raise InvalidCursorError(refused)
    return stored, pointer, offset
