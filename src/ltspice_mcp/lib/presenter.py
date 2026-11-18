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
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

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
class Presented:
    """The reply to send, the lists it cut, and the stored answer behind it."""

    reply: dict[str, Any]
    cut: dict[str, ListCut]
    stored: StoredAnswer | None


#: Adds the tool's own words about a cut to a reply: ``(reply, cut, stored,
#: paged)``, where ``paged`` is the pointer of the list a page reads, or None
#: on a first reply. Called on each candidate, so what it adds is measured.
Annotate = Callable[[dict[str, Any], Mapping[str, ListCut], StoredAnswer, str | None], None]


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


def _size(reply: Mapping[str, Any]) -> int:
    return estimate_tokens(reply, as_sent=False)


def _fit(
    sources: Mapping[str, list[Any]],
    build: Callable[[Mapping[str, int]], dict[str, Any]],
    target: int,
) -> dict[str, int]:
    """How many entries of each list in ``sources`` the reply keeps.

    The largest list still holding more than one entry is cut first, to the
    most entries that let the reply fit; when none does, to one entry, and the
    next largest is cut.
    """
    keep = {pointer: len(entries) for pointer, entries in sources.items()}
    if _size(build(keep)) <= target:
        return keep
    while True:
        open_lists = [pointer for pointer, count in keep.items() if count > 1]
        if not open_lists:
            return keep
        largest = max(
            open_lists,
            key=lambda p: len(json.dumps(sources[p][: keep[p]], default=str)),
        )
        low, high = 1, keep[largest] - 1
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            if _size(build({**keep, largest: middle})) <= target:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best is not None:
            keep[largest] = best
            return keep
        keep[largest] = 1


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


def _expired(data: Mapping[str, Any]) -> bool:
    expires = parse_iso_datetime(str(data.get("expires_at", "")))
    return expires is None or expires <= now()


def cleanup(store: Store) -> int:
    """Delete the stored answers past their expiry; how many were deleted."""
    removed = 0
    try:
        paths = list(store.results_dir.glob(f"{STORED_PREFIX}*.json"))
    except OSError:
        return 0
    for path in paths:
        try:
            data = json.loads(read_text_file(path))
            if not isinstance(data, dict) or _expired(data):
                path.unlink()
                removed += 1
        except (OSError, ValueError):
            continue
    return removed


def _store_answer(
    store: Store,
    tool: str,
    answer: dict[str, Any],
    snapshot: dict[str, Any],
    ttl_hours: float,
) -> StoredAnswer:
    cleanup(store)
    answer_id = _answer_id(tool, answer, snapshot)
    path = store.stored_answer(answer_id)
    created = now()
    store.ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        envelope(
            KIND_STORED_ANSWER,
            id=answer_id,
            tool=tool,
            created_at=created.isoformat(),
            expires_at=(created + timedelta(hours=ttl_hours)).isoformat(),
            snapshot=snapshot,
            answer=answer,
        ),
    )
    return StoredAnswer(answer_id, path, tool, snapshot, answer)


def _cursor(answer_id: str, pointer: str, offset: int) -> str:
    return _encode_body({"answer": answer_id, "list": pointer, "offset": offset})


# ---------------------------------------------------------------------------
# Presenting
# ---------------------------------------------------------------------------


def _present_lists(
    base: dict[str, Any],
    sources: Mapping[str, list[Any]],
    offsets: Mapping[str, int],
    totals: Mapping[str, int],
    stored: StoredAnswer,
    *,
    target: int,
    annotate: Annotate | None,
    paged: str | None,
    fixed: Mapping[str, ListCut] | None = None,
) -> Presented:
    """``base`` with each list of ``sources`` cut to fit ``target``. A list
    is named in ``cut`` when the reply holds less than all of it, or when it
    is the list a page reads (``paged``); ``fixed`` names lists the reply
    leaves out whole."""

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
                next=_cursor(stored.id, pointer, end) if end < total else None,
            )
        made.update(fixed or {})
        return made

    def build(keep: Mapping[str, int]) -> dict[str, Any]:
        reply = _with_lists(
            base,
            {pointer: entries[: keep[pointer]] or None for pointer, entries in sources.items()},
        )
        made = cuts(keep)
        reply["cut"] = {pointer: entry.as_reply() for pointer, entry in made.items()}
        reply["stored_answer"] = stored.as_reply()
        if annotate is not None:
            annotate(reply, made, stored, paged)
        return reply

    keep = _fit(sources, build, target)
    return Presented(build(keep), cuts(keep), stored)


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
) -> Presented:
    """Send ``answer`` whole when it fits ``target`` tokens, or store it and
    cut its cuttable lists to fit.

    ``answer`` is the reply as ``reply_payload`` makes it. A ``target`` of
    None sends every answer whole. ``snapshot`` may be a function, called
    only when the answer is stored, so an answer that fits reads nothing.
    """
    if target is None or _size(answer) <= target:
        return Presented(answer, {}, None)
    sources = {
        pointer: resolve_pointer(answer, pointer)
        for pointer, _ in cuttable_lists(answer, schema)
        if len(resolve_pointer(answer, pointer)) > 1
    }
    if not sources:
        return Presented(answer, {}, None)
    stored = _store_answer(
        store, tool, answer, snapshot() if callable(snapshot) else snapshot, ttl_hours
    )
    return _present_lists(
        answer,
        sources,
        {},
        {pointer: len(entries) for pointer, entries in sources.items()},
        stored,
        target=target,
        annotate=annotate,
        paged=None,
    )


def present_page(
    stored: StoredAnswer,
    pointer: str,
    offset: int,
    schema: Mapping[str, Any],
    *,
    target: int | None,
    annotate: Annotate | None = None,
) -> Presented:
    """The entries of the list at ``pointer`` from ``offset`` on, read from
    ``stored`` and cut to fit ``target``.

    The page carries the stored answer's other fields as they are, but none of
    its other cuttable lists: each is named in ``cut`` with none returned and
    the cursor that reads it from the start.
    """
    answer = stored.answer
    whole = resolve_pointer(answer, pointer)
    others = {
        other: resolve_pointer(answer, other)
        for other, _ in cuttable_lists(answer, schema)
        if other != pointer
    }
    return _present_lists(
        _with_lists(answer, dict.fromkeys(others)),
        {pointer: whole[offset:]},
        {pointer: offset},
        {pointer: len(whole)},
        stored,
        target=sys.maxsize if target is None else target,
        annotate=annotate,
        paged=pointer,
        fixed={
            other: ListCut(total=len(entries), returned=0, next=_cursor(stored.id, other, 0))
            for other, entries in others.items()
        },
    )


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
    refused = (
        f"{text[:40]!r} is not a cursor this tool returned: pass a cut entry's next unchanged."
    )
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
