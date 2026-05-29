"""Cross-card transformation passes over ``list[SpiceCard]``.

Layer 2 typed views (``ModelCard``, ``InstanceLine``, etc.) handle
edits that fit inside one card. Operations that touch multiple cards
atomically — renaming a subcircuit (opener + closer + every body
card's scope), renaming a model (``.MODEL`` + every instance
referencing it), injecting cards into the flat list — live here.

Public surface:

- ``inject_card_before_end(cards, text, scope)`` — insert a parsed
  card before the top-level ``.END``. Falls back to appending when no
  ``.END`` is present.
- ``rename_subckt(cards, old_name, new_name)`` — rename a subcircuit
  across its opener, matching ``.ENDS``, every body card's
  ``scope`` tuple, and every ``Xxxx`` invocation that calls it.
- ``rename_model(cards, old_name, new_name, scope)`` — rename a
  ``.MODEL`` and every ``Mxxx`` / ``Qxxx`` / ``Jxxx`` reference visible
  from ``scope``. Scope-aware: instances in a different scope that
  reference an outer-scope model are still updated.

Future cross-card transformations (component rename, subcircuit
inline/extract, structural diff, atomic change-set commit) will land
here as concrete implementations rather than stubs — see the foundation
plan for the roadmap.

Conventions:

- Functions take a mutable ``list[SpiceCard]`` and modify it in place.
  Cards inside the list are mutated through ``replace_span`` /
  ``replace_body`` so ``emit()`` re-renders correctly.
- Functions raise ``ValueError`` (or a subclass like ``SpiceLexError``)
  on invalid input; partial mutations are not rolled back. Callers
  that need atomic commit should snapshot ``cards`` first.
"""

from __future__ import annotations

from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    SpiceLexErrorCategory,
    find_matching_ends,
    lex,
)
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    ModelCard,
    SubcktCard,
)

# ---------------------------------------------------------------------------
# Card injection
# ---------------------------------------------------------------------------


def inject_card_before_end(
    cards: list[SpiceCard],
    text: str,
    *,
    scope: tuple[str, ...] = (),
) -> SpiceCard:
    """Insert a parsed card before the top-level ``.END`` directive.

    ``text`` is parsed via ``lex`` and must produce exactly one card
    (plus optionally a trailing newline as a blank). Multi-card text
    raises ``ValueError``. The returned card has been added to
    ``cards`` and carries the requested ``scope``.

    If no ``.END`` is present at top level, the card is appended.
    Either way, the card preceding the insertion point is patched
    with a trailing newline if it lacks one — without that fix,
    ``emit`` concatenates two cards' raw_lines on the same line and
    produces a malformed netlist (e.g. ``R1 a b 1k.MODEL X NMOS``).
    Replaces the legacy ``montecarlo.inject_card_before_end``.
    """
    sub = lex(text).cards
    real_cards = [c for c in sub if c.kind not in ("blank", "comment")]
    if len(real_cards) != 1:
        raise ValueError(
            f"inject_card_before_end: text must parse to exactly one real card, "
            f"got {len(real_cards)}: {text!r}"
        )
    new_card = real_cards[0]
    new_card.scope = scope
    # Find the top-level .END (scope=()) and insert before it.
    for i, c in enumerate(cards):
        if c.kind == "end" and c.scope == ():
            _ensure_predecessor_ends_with_newline(cards, i)
            cards.insert(i, new_card)
            return new_card
    # No .END — append.
    _ensure_predecessor_ends_with_newline(cards, len(cards))
    cards.append(new_card)
    return new_card


def _ensure_predecessor_ends_with_newline(cards: list[SpiceCard], insert_idx: int) -> None:
    """Patch ``cards[insert_idx - 1]``'s last raw_line so it ends with a newline.

    No-op when there is no predecessor (insert at the start) or when
    the predecessor already ends with ``\\n``/``\\r\\n``. Modifies
    ``raw_lines`` in place; doesn't flip ``dirty`` because adding a
    final newline isn't a semantic edit — it's correcting an
    end-of-file irregularity that would otherwise cause ``emit`` to
    glue two cards together.
    """
    if insert_idx <= 0:
        return
    prev = cards[insert_idx - 1]
    if not prev.raw_lines:
        return
    last = prev.raw_lines[-1]
    if last.endswith("\n"):
        return
    prev.raw_lines = [*prev.raw_lines[:-1], last + "\n"]


# ---------------------------------------------------------------------------
# Subcircuit rename
# ---------------------------------------------------------------------------


def rename_subckt(
    cards: list[SpiceCard],
    old_name: str,
    new_name: str,
) -> int:
    """Rename a subcircuit across opener, closer, body, and callers.

    **Validate-then-commit semantics.** First pass walks the card list
    and gathers all targets (opener, optional matching ``.ENDS``, every
    ``Xxxx`` invocation, every body card whose scope includes
    ``old_name``). Each ``Xxxx`` is parsed via ``InstanceLine.from_card``
    during validation; if any parse raises, ``rename_subckt`` raises
    before any mutation lands. Second pass commits the gathered edits.

    This is **not** a true atomic transaction: validation rules out the
    common failure paths (malformed Xxxx lines, missing opener) but if
    a commit-time mutation raises despite validation — e.g. a card was
    mutated externally between the two passes — the netlist is left
    partially renamed. Callers needing strict atomicity should
    snapshot ``cards`` first.

    Updates on commit:

    - ``.SUBCKT <old> ...`` opener card's name.
    - Matching ``.ENDS [<old>]`` closer card's trailing name (if named).
    - ``scope`` tuple of every body card whose scope contains ``old_name``.
    - Every ``Xxxx`` invocation whose model/subckt token is ``old_name``.

    Returns the number of cards modified. Raises ``SpiceLexError`` of
    category ``MALFORMED_CARD`` if no matching opener is found or if
    any ``Xxxx`` candidate fails to parse. Comparisons are
    case-insensitive (SPICE convention); the new name is written verbatim.
    """
    target = old_name.lower()

    # ---- Validation pass: locate opener, closer, X callers, scope updates.
    opener: SpiceCard | None = None
    for c in cards:
        if c.kind == "subckt" and c.name and c.name.lower() == target:
            opener = c
            break
    if opener is None:
        raise SpiceLexError(
            SpiceLexErrorCategory.MALFORMED_CARD,
            f"rename_subckt: no .SUBCKT named {old_name!r} found",
        )

    opener_idx = cards.index(opener)
    closer_idx = find_matching_ends(cards, opener_idx)
    closer = cards[closer_idx] if closer_idx is not None else None

    x_callers: list[tuple[SpiceCard, InstanceLine]] = []
    for c in cards:
        if c.kind != "instance":
            continue
        if not c.name or c.name[:1].upper() != "X":
            continue
        try:
            view = InstanceLine.from_card(c)
        except SpiceLexError as e:
            raise SpiceLexError(
                SpiceLexErrorCategory.MALFORMED_CARD,
                f"rename_subckt: failed to parse X invocation {c.name!r} "
                f"at line {c.line_start}: {e}",
            ) from e
        if view.model and view.model.lower() == target:
            x_callers.append((c, view))

    scope_updates: list[tuple[SpiceCard, tuple[str, ...]]] = []
    for c in cards:
        if not c.scope:
            continue
        if any(s.lower() == target for s in c.scope):
            new_scope = tuple(new_name if s.lower() == target else s for s in c.scope)
            if new_scope != c.scope:
                scope_updates.append((c, new_scope))

    # ---- Commit pass.
    n_modified = 0
    SubcktCard.from_card(opener).set_name_local(new_name)
    opener.name = new_name
    n_modified += 1

    if closer is not None and closer.name and closer.name.lower() == target:
        closer.replace_body(f".ENDS {new_name}")
        closer.name = new_name
        n_modified += 1

    for c, new_scope in scope_updates:
        c.scope = new_scope
        n_modified += 1

    for _c, view in x_callers:
        view.set_model(new_name)
        n_modified += 1

    return n_modified


# ---------------------------------------------------------------------------
# Model rename
# ---------------------------------------------------------------------------


def rename_model(
    cards: list[SpiceCard],
    old_name: str,
    new_name: str,
    *,
    scope: tuple[str, ...] = (),
) -> int:
    """Rename a ``.MODEL`` card and every M/Q/J reference visible from ``scope``.

    Walks the card list and:

    - Renames every ``.MODEL <old>`` card whose scope is ``scope``.
    - Renames every ``Mxxx`` / ``Qxxx`` / ``Jxxx`` instance whose
      ``model`` is ``old_name`` and whose scope is ``scope`` or a
      strict descendant (SPICE name-resolution: inner scopes see outer
      models).

    Returns the count of cards modified. Comparisons are
    case-insensitive.
    """
    target = old_name.lower()
    n_modified = 0

    for c in cards:
        if c.kind == "model" and c.name and c.name.lower() == target and c.scope == scope:
            ModelCard.from_card(c).set_name(new_name)
            n_modified += 1
        elif c.kind == "instance":
            if not c.name:
                continue
            prefix = c.name[:1].upper()
            if prefix not in ("M", "Q", "J"):
                continue
            # Scope check: instance scope must be ``scope`` or extend it.
            if len(c.scope) < len(scope) or c.scope[: len(scope)] != scope:
                continue
            view = InstanceLine.from_card(c)
            if view.model and view.model.lower() == target:
                view.set_model(new_name)
                n_modified += 1

    return n_modified
