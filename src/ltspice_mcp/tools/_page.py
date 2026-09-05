"""One offset-paged envelope shape.

Every paginated surface built on this returns the same five keys — ``items``,
``total``, ``returned``, ``truncated``, ``next_cursor`` — so a caller learns the
shape once and continues any of them the same way. ``next_cursor`` is always
present and nullable: omitting it on the last page makes an unconditional
``page["next_cursor"]`` raise ``KeyError`` instead of reading ``None``.

The cursor grammar is ``o:<offset>``. DECODING one is the caller's job, because
the error a malformed cursor should produce belongs to the tool the caller
addressed, not to this shape.

Two entry points, differing only in who did the slicing: ``page`` takes the
whole list and slices it; ``page_of`` takes a window a caller already cut, for
the surfaces that project rows AFTER paging so the projection runs only over
the rows the page carries. ``unpaged`` is the whole-result form the in-process
door returns.

The checksummed, binding-scoped cursor in ``lib/pin_legend`` is a different
contract — it refuses a cursor minted against another query or another revision
of the file — and stays separate.
"""

from __future__ import annotations

from typing import Any

#: The offset-cursor grammar: an ``o:`` tag and a decimal offset.
CURSOR_PREFIX = "o"


def encode_offset_cursor(offset: int) -> str:
    """The cursor that continues a page at ``offset``."""
    return f"{CURSOR_PREFIX}:{offset}"


def page_of(window: list[Any], *, offset: int, total: int) -> dict[str, Any]:
    """Report an already-sliced ``window`` starting at ``offset`` as one page."""
    end = offset + len(window)
    return {
        "items": window,
        "total": total,
        "returned": len(window),
        "truncated": end < total,
        "next_cursor": encode_offset_cursor(end) if end < total else None,
    }


def page(items: list[Any], *, offset: int = 0, limit: int) -> dict[str, Any]:
    """One page of ``items`` starting at ``offset``.

    ``offset`` is clamped to the end of ``items``, so a cursor left over from a
    shorter list yields an empty last page rather than raising. ``limit`` is
    floored at 1: a zero limit would report ``truncated`` with a cursor pointing
    back at the same offset — a pagination loop that never advances.
    """
    start = min(max(offset, 0), len(items))
    return page_of(items[start : start + max(1, limit)], offset=start, total=len(items))


def unpaged(items: list[Any]) -> dict[str, Any]:
    """The same envelope with every item included and nothing left to continue."""
    return page_of(items, offset=0, total=len(items))
