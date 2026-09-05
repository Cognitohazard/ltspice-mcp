"""One offset-paged envelope shape.

Every paginated surface built on this returns the same five keys — ``items``,
``total``, ``returned``, ``truncated``, ``next_cursor`` — so a caller learns the
shape once and continues any of them the same way. ``next_cursor`` is always
present and nullable: omitting it on the last page makes an unconditional
``page["next_cursor"]`` raise ``KeyError`` instead of reading ``None``.

The cursor grammar is ``o:<offset>``. Decoding one is the caller's job, because
the error a malformed cursor should produce belongs to the tool the caller
addressed, not to this shape.
"""

from __future__ import annotations

from typing import Any

#: The offset-cursor grammar: an ``o:`` tag and a decimal offset.
CURSOR_PREFIX = "o"


def encode_offset_cursor(offset: int) -> str:
    """The cursor that continues a page at ``offset``."""
    return f"{CURSOR_PREFIX}:{offset}"


def page(
    items: list[Any],
    *,
    offset: int = 0,
    limit: int,
) -> dict[str, Any]:
    """One page of ``items`` starting at ``offset``, in the shared envelope.

    ``offset`` is clamped to the end of ``items``, so a cursor left over from a
    shorter list yields an empty last page rather than raising.
    """
    start = min(max(offset, 0), len(items))
    window = items[start : start + limit]
    truncated = start + len(window) < len(items)
    return {
        "items": window,
        "total": len(items),
        "returned": len(window),
        "truncated": truncated,
        "next_cursor": encode_offset_cursor(start + len(window)) if truncated else None,
    }


def unpaged(items: list[Any]) -> dict[str, Any]:
    """The same envelope with every item included and nothing left to continue."""
    return {
        "items": items,
        "total": len(items),
        "returned": len(items),
        "truncated": False,
        "next_cursor": None,
    }
