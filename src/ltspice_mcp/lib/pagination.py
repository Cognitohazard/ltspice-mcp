"""The page shape every list surface returns, and the offset cursor form.

One page is ``{items, total, returned, truncated, next_cursor}``. ``next_cursor``
is always present and nullable so a reader may write ``page["next_cursor"]``
unconditionally — omitting the key on the last page instead turns the end of a
listing into a ``KeyError`` for exactly the caller who was looping correctly.

Two cursor grammars reach these pages. A surface that resumes into immutable
stored work mints a checksummed cursor of its own (``lib/cursor_codec``, via
``result_store``), because the position it encodes is meaningless without the
set it was minted against. A surface paging a list it recomputes each call uses
the plain ``"o:<offset>"`` form here: there is nothing to tamper with beyond an
offset, and a bad one is refused by :func:`decode_offset` rather than silently
clamped to zero, which would restart a listing the caller believed it was
continuing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def encode_offset(offset: int) -> str:
    """The cursor naming the row a listing resumes at."""
    return f"o:{offset}"


def decode_offset(cursor: str | None) -> int | None:
    """The offset an ``"o:<offset>"`` cursor names; ``None`` if it is not one.

    Absent means the first page, so ``None`` in gives 0 out. A malformed cursor
    gives ``None`` back rather than raising, because the error a caller should
    report depends on the surface it came in on.
    """
    if cursor is None:
        return 0
    prefix, separator, raw_offset = cursor.partition(":")
    if separator != ":" or prefix != "o" or not raw_offset.isdecimal():
        return None
    return int(raw_offset)


def page(
    items: list[Any],
    offset: int,
    limit: int,
    *,
    cursor: Callable[[int], str] | None = None,
) -> tuple[dict[str, Any], int]:
    """One page of ``items``, plus the offset the page after it starts at.

    ``cursor`` builds the next cursor from that offset, and is called only when
    there is a next page. A surface whose cursor cannot be minted until later —
    an analysis page, whose cursor also has to carry the work position — leaves
    it out and fills ``next_cursor`` in itself from the returned offset, so
    where the next page begins is still decided in one place.
    """
    empty: dict[str, Any] = {
        "items": [],
        "total": len(items),
        "returned": 0,
        "truncated": False,
        "next_cursor": None,
    }
    next_offset = retotal_page(empty, items[offset : offset + limit], offset)
    if empty["truncated"] and cursor is not None:
        empty["next_cursor"] = cursor(next_offset)
    return empty, next_offset


def retotal_page(page_data: dict[str, Any], rows: list[Any], offset: int) -> int:
    """Replace a page's rows and reconcile its metadata; return the next offset.

    ``total`` is left alone: it counts what the caller asked about, and rows
    that were projected, rendered or dropped for the response budget do not
    change how many there were.
    """
    next_offset = offset + len(rows)
    page_data.update(
        items=rows,
        returned=len(rows),
        truncated=next_offset < page_data["total"],
        next_cursor=None,
    )
    return next_offset
