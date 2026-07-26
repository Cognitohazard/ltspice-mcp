"""Pin legend + label-only-pin views over a placed schematic.

A *pin legend* names, per component, every pin and the net it sits on — the
compact "what connects to what" table a model reads back after an edit. A
*label-only pin* is one whose only tie to its net is a net-label at its
coordinate (no wire runs through it): the net-label-soup signature that
simulates correctly but reads as a wiring list rather than a drawn schematic.

This module holds only the assembly, classification, and pagination logic. The
connectivity itself — which net a coordinate is on, whether a wire passes
through it — is resolved by the caller through the shared ``_net_partition`` /
``_build_on_wire_predicate`` machinery in ``tools/circuit.py`` and handed in as
plain data + closures, so this stays a pure leaf module (no ``tools`` import,
no re-implementation of the union-find).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ltspice_mcp.lib.cursor_codec import CursorError, decode_cursor, encode_cursor

# Default page size for the paginated views. Kept local (not imported from the
# tool layer) so this module has no upward dependency.
DEFAULT_PAGE_SIZE = 100

# Coordinate → its net's display name (a net-label like ``vout``/``0``, or
# ``None`` when the net is unnamed). Built by the caller from ``_net_partition``.
NetNameOf = Callable[["tuple[int, int]"], "str | None"]
# Coordinate → whether a wire passes through it, and whether a net-label sits on
# it. Built by the caller from ``_build_on_wire_predicate`` and the label set.
CoordPredicate = Callable[["tuple[int, int]"], bool]


def build_pin_legend(
    geometry: Sequence[Mapping[str, Any]],
    net_name_of: NetNameOf,
) -> list[dict[str, Any]]:
    """Per-component legend: ``[{ref, pins: [{name, x, y, dir?, net}]}]``.

    ``geometry`` is ``_collect_component_geometry``'s output (components with
    resolvable symbol geometry, each with a ``pins`` list). ``net_name_of``
    resolves a pin coordinate to its net name (``None`` = unnamed net).
    """
    legend: list[dict[str, Any]] = []
    for comp in geometry:
        pins: list[dict[str, Any]] = []
        for p in comp["pins"]:
            coord = (p["x"], p["y"])
            entry: dict[str, Any] = {
                "name": p["name"],
                "x": p["x"],
                "y": p["y"],
                "net": net_name_of(coord),
            }
            if p.get("dir") is not None:
                entry["dir"] = p["dir"]
            pins.append(entry)
        legend.append({"ref": comp["ref"], "pins": pins})
    return legend


def find_label_only_pins(
    geometry: Sequence[Mapping[str, Any]],
    *,
    is_wired: CoordPredicate,
    is_labeled: CoordPredicate,
    net_name_of: NetNameOf,
) -> list[dict[str, Any]]:
    """Pins connected to their net by a net-label only (no wire through them).

    Same classification as the wiring metric's ``pins_label_only`` (a pin is
    wired when a wire passes through it, else label-only when a flag sits on
    it), returned as an addressable list: ``[{ref, pin, net, x, y}]``.
    """
    out: list[dict[str, Any]] = []
    for comp in geometry:
        ref = comp["ref"]
        for p in comp["pins"]:
            coord = (p["x"], p["y"])
            if is_wired(coord) or not is_labeled(coord):
                continue
            out.append(
                {
                    "ref": ref,
                    "pin": f"{ref}.{p['name']}",
                    "net": net_name_of(coord),
                    "x": p["x"],
                    "y": p["y"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Pagination — opaque, tamper-evident, kind-scoped cursors
# ---------------------------------------------------------------------------


class PageCursorError(ValueError):
    """A resumption cursor did not decode, or was minted for another view."""


def encode_page_cursor(kind: str, offset: int) -> str:
    """Opaque cursor naming the view (``kind``) and the resume ``offset``.

    ``kind`` binds the cursor to one view so a ``pin_legend`` cursor cannot be
    replayed against ``label_only_pins`` (the two are different lists). Integrity
    is the shared codec's checksum; the ``kind`` field is the functional view
    binding on top of it.
    """
    return encode_cursor({"kind": kind, "offset": offset})


def decode_page_cursor(cursor: str, kind: str) -> int:
    """Recover the resume offset from ``cursor``; validate it belongs to ``kind``."""
    try:
        body = decode_cursor(cursor)
        c_kind = body["kind"]
        offset = int(body["offset"])
    except (CursorError, KeyError, TypeError, ValueError) as exc:
        raise PageCursorError(f"malformed page cursor: {cursor!r}") from exc
    if c_kind != kind:
        raise PageCursorError(
            f"cursor is for view {c_kind!r}, not {kind!r}; use the page's own next_cursor"
        )
    if offset < 0:
        raise PageCursorError(f"invalid page cursor: {cursor!r}")
    return offset


def encode_pair_cursor(kind: str, offsets: tuple[int, int]) -> str:
    """Opaque cursor for a view paging TWO collections under one token.

    The body key differs from the single-offset cursor's (``offsets`` vs
    ``offset``), so handing either decoder the other's cursor raises rather than
    silently dropping an offset — a dropped second offset would re-serve the
    same rows forever.
    """
    return encode_cursor({"kind": kind, "offsets": [offsets[0], offsets[1]]})


def decode_pair_cursor(cursor: str, kind: str) -> tuple[int, int]:
    """Recover both resume offsets from ``cursor``; validate it belongs to ``kind``."""
    try:
        body = decode_cursor(cursor)
        c_kind = body["kind"]
        first, second = (int(value) for value in body["offsets"])
    except (CursorError, KeyError, TypeError, ValueError) as exc:
        raise PageCursorError(f"malformed page cursor: {cursor!r}") from exc
    if c_kind != kind:
        raise PageCursorError(
            f"cursor is for view {c_kind!r}, not {kind!r}; use the page's own next_cursor"
        )
    if first < 0 or second < 0:
        raise PageCursorError(f"invalid page cursor: {cursor!r}")
    return first, second


def _window(items: Sequence[Any], offset: int, limit: int) -> dict[str, Any]:
    """One page of ``items`` from ``offset``, plus ``end`` — where the next starts.

    The single slicing rule behind every paginated view, so a page's contents
    and the cursor that resumes it can never disagree.
    """
    total = len(items)
    offset = min(max(0, offset), total)
    limit = max(1, int(limit))
    window = list(items[offset : offset + limit])
    end = offset + len(window)
    return {
        "items": window,
        "total": total,
        "returned": len(window),
        "end": end,
        "truncated": end < total,
    }


def paginate_view(
    items: Sequence[Any],
    kind: str,
    *,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Slice ``items`` into a ``{items, total, returned, truncated, next_cursor}``
    page. ``cursor`` resumes a prior page (its offset must belong to ``kind``).

    ``primary_truncated`` repeats ``truncated`` so both paginators report each
    collection's own exhaustion beside it under one name: a reader of either
    page can ask "does THIS collection continue?" without first knowing which
    paginator produced it.
    """
    offset = decode_page_cursor(cursor, kind) if cursor else 0
    window = _window(items, offset, limit)
    return {
        "items": window["items"],
        "total": window["total"],
        "returned": window["returned"],
        "truncated": window["truncated"],
        "primary_truncated": window["truncated"],
        "next_cursor": encode_page_cursor(kind, window["end"]) if window["truncated"] else None,
    }


def paginate_pair(
    primary: Sequence[Any],
    secondary: Sequence[Any],
    kind: str,
    *,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    secondary_limit: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Page two collections of ONE item under a single resumption token.

    Each collection advances by its own offset — both ride in the cursor — so
    successive pages repeat neither, and neither is capped at its first page.
    ``truncated`` and ``next_cursor`` cover the pair: a cursor is minted while
    *either* collection has more, which keeps "page until next_cursor is null"
    the correct stop rule. Each collection's own exhaustion is reported beside it
    (``primary_truncated`` / ``secondary_truncated``) so the caller can tell which
    one continues — the pair-level ``truncated`` cannot say.
    """
    first_offset, second_offset = decode_pair_cursor(cursor, kind) if cursor else (0, 0)
    first = _window(primary, first_offset, limit)
    second = _window(secondary, second_offset, secondary_limit)
    truncated = first["truncated"] or second["truncated"]
    return {
        "items": first["items"],
        "total": first["total"],
        "returned": first["returned"],
        "primary_truncated": first["truncated"],
        "secondary_items": second["items"],
        "secondary_total": second["total"],
        "secondary_returned": second["returned"],
        "secondary_truncated": second["truncated"],
        "truncated": truncated,
        "next_cursor": (
            encode_pair_cursor(kind, (first["end"], second["end"])) if truncated else None
        ),
    }
