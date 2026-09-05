"""The page shape every list surface returns, and the offset cursor form.

One page is ``{items, total, returned, truncated, next_cursor}``, so a caller
learns the shape once and continues any surface the same way. ``next_cursor``
is always present and nullable: omitting it on the last page turns the end of a
listing into a ``KeyError`` for exactly the caller who was looping correctly.

Two cursor grammars reach these pages. A surface that resumes into immutable
stored work mints a checksummed cursor of its own (``lib/cursor_codec``, via
``result_store``), because the position it encodes is meaningless without the
set it was minted against — and the analysis pages go further, carrying a work
position alongside the row offset, so they pass ``cursor=None`` here and fill
the key in themselves. A surface paging a list it recomputes each call uses the
plain ``"o:<offset>"`` form: there is nothing to tamper with beyond an offset.

Decoding one is the caller's job — :func:`decode_offset` reports a malformed
cursor as ``None`` rather than raising, because the error a caller should see
belongs to the tool it addressed, and because a silent clamp to zero would
restart a listing the caller believed it was continuing.

Three entry points, differing in who did the slicing: :func:`page` takes the
whole list and slices it, with the guards a caller-supplied offset and limit
need; :func:`page_of` takes a window a caller already cut, for the surfaces
that project rows AFTER paging so the projection runs only over the rows the
page carries; :func:`unpaged` is the whole-result form the Python API
returns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: The offset-cursor grammar: an ``o:`` tag and a decimal offset.
CURSOR_PREFIX = "o"

#: Builds the next cursor from the offset the next page starts at. ``None``
#: leaves ``next_cursor`` null for a surface that mints its own.
CursorFor = Callable[[int], str] | None


def encode_offset(offset: int) -> str:
    """The cursor that continues a listing at ``offset``."""
    return f"{CURSOR_PREFIX}:{offset}"


def decode_offset(cursor: str | None) -> int | None:
    """The offset an ``"o:<offset>"`` cursor names; ``None`` if it is not one.

    Absent means the first page, so ``None`` in gives 0 out.
    """
    if cursor is None:
        return 0
    prefix, separator, raw_offset = cursor.partition(":")
    if separator != ":" or prefix != CURSOR_PREFIX or not raw_offset.isdecimal():
        return None
    return int(raw_offset)


def page_of(
    window: list[Any],
    *,
    offset: int,
    total: int,
    cursor: CursorFor = encode_offset,
) -> dict[str, Any]:
    """Report an already-sliced ``window`` starting at ``offset`` as one page."""
    end = offset + len(window)
    truncated = end < total
    return {
        "items": window,
        "total": total,
        "returned": len(window),
        "truncated": truncated,
        "next_cursor": cursor(end) if truncated and cursor is not None else None,
    }


def page(
    items: list[Any],
    *,
    offset: int = 0,
    limit: int,
    cursor: CursorFor = encode_offset,
) -> dict[str, Any]:
    """One page of ``items`` starting at ``offset``.

    ``offset`` is clamped to the end of ``items``, so a cursor left over from a
    shorter list yields an empty last page rather than raising. ``limit`` is
    floored at 1: a zero limit would report ``truncated`` with a cursor pointing
    back at the same offset — a pagination loop that never advances.
    """
    start = min(max(offset, 0), len(items))
    return page_of(
        items[start : start + max(1, limit)], offset=start, total=len(items), cursor=cursor
    )


def unpaged(items: list[Any]) -> dict[str, Any]:
    """The same envelope with every item included and nothing left to continue."""
    return page_of(items, offset=0, total=len(items))


def retotal_page(page_data: dict[str, Any], rows: list[Any], offset: int) -> int:
    """Replace a page's rows and reconcile its metadata; return the next offset.

    For a surface that renders or projects its rows after paging. ``total`` is
    left alone: it counts what the caller asked about, and rows that were
    projected, rendered or dropped for the response budget do not change how
    many there were. ``next_cursor`` is cleared, because the caller that
    re-rowed the page is the one that decides what continues it.
    """
    next_offset = offset + len(rows)
    page_data.update(
        items=rows,
        returned=len(rows),
        truncated=next_offset < page_data["total"],
        next_cursor=None,
    )
    return next_offset
