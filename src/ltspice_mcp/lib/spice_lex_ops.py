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
- ``value_suffix_sites(cards)`` — every number whose scale-suffix
  position holds a non-ASCII character (``23µ``, ``23Âµ``), outside
  comments, ``.control`` blocks, include paths and double-quoted strings.
- ``fold_micro_suffix_cards(cards)`` — the same scan, with each micro sign
  found at a suffix position rewritten as ``u`` in place.
- ``strip_instance_section_signs(cards)`` — drop the ``§`` LTspice's
  netlister writes into an instance name, from that name wherever the deck
  names it, and nowhere else.

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

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ltspice_mcp.lib.format import MICRO_SIGNS
from ltspice_mcp.lib.spice_lex import (
    INCLUDE_HEADS,
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
    inject_cards_before_end(cards, [new_card])
    return new_card


def inject_cards_before_end(cards: list[SpiceCard], block: list[SpiceCard]) -> None:
    """Splice a block of already-parsed cards in before the top-level ``.END``.

    The multi-card form of ``inject_card_before_end``, for a caller that has
    built its cards itself — a whole ``.SUBCKT`` body, say — and would only be
    re-lexing them to hand over text. Same newline handling: both the block's
    last line and the line it lands after are given a trailing newline if they
    lack one, or ``emit`` glues two cards onto one line.

    ``block`` is spliced in as given, so the caller's card objects (and any
    scope they carry) are what ends up in ``cards``.
    """
    if not block:
        return
    # Guard the block's OWN trailing newline: when it lands before a .END,
    # emit would otherwise glue ".END" onto its last line (the predecessor
    # guard below only fixes the line before the insertion point).
    last = block[-1]
    if last.raw_lines and not last.raw_lines[-1].endswith("\n"):
        last.raw_lines = [*last.raw_lines[:-1], last.raw_lines[-1] + "\n"]
    # Find the top-level .END (scope=()) and insert before it; no .END, append.
    at = next((i for i, c in enumerate(cards) if c.kind == "end" and c.scope == ()), len(cards))
    _ensure_predecessor_ends_with_newline(cards, at)
    cards[at:at] = block


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


# ---------------------------------------------------------------------------
# Non-ASCII value suffixes
# ---------------------------------------------------------------------------

# A number at a token boundary followed directly by a non-ASCII character: the
# character sits where a scale suffix goes. The lookbehind keeps a digit run
# inside a name (``N001µ``, ``x1µ``) out, and one after a backslash, which is a
# Windows path separator and never an operator. Group 3 is the rest of the
# token, up to whitespace, a delimiter or an operator.
_SUFFIX_SITE_RE = re.compile(
    r"(?<![\w.\\])((?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)([^\x00-\x7f])"
    r"([^\s=(){}\[\],\"';$*/+\-<>!&|^?:%]*)"
)


@dataclass(frozen=True)
class ValueSuffixSite:
    """A number whose scale-suffix position holds a non-ASCII character.

    ``offset`` is the body offset of that character in ``card``; ``line`` is
    the 1-based source line it sits on. ``number`` is the mantissa as written,
    ``suffix`` the character, and ``tail`` the rest of the token after it.
    """

    card: SpiceCard = field(compare=False, repr=False)
    offset: int
    line: int
    number: str
    suffix: str
    tail: str

    @property
    def token(self) -> str:
        """The whole token as written (``23Âµ``, ``4.7µF``)."""
        return self.number + self.suffix + self.tail

    @property
    def micro(self) -> bool:
        """The character is a micro sign, so a reader that knows it reads 1e-6."""
        return self.suffix in MICRO_SIGNS

    @property
    def misdecoded_micro(self) -> bool:
        """The suffix is a UTF-8 micro sign decoded as cp1252 (``Âµ``, ``Î¼``)."""
        try:
            return (self.suffix + self.tail[:1]).encode("cp1252").decode("utf-8") in MICRO_SIGNS
        except UnicodeError:
            return False


def _unquoted_spans(body: str) -> Iterator[tuple[int, int]]:
    """``(start, end)`` of every stretch of ``body`` outside double quotes.

    A double-quoted string is a file name or a label, never a value. An
    unterminated quote runs to the end of the body.
    """
    start = 0
    while True:
        opening = body.find('"', start)
        if opening < 0:
            yield start, len(body)
            return
        yield start, opening
        closing = body.find('"', opening + 1)
        if closing < 0:
            return
        start = closing + 1


def value_suffix_sites(cards: Iterable[SpiceCard]) -> list[ValueSuffixSite]:
    """Every number in ``cards`` whose suffix position holds a non-ASCII character.

    Comments never reach a card body, and a ``.control`` block has none, so
    neither is scanned. Include-family paths, where a digit-then-µ run is part
    of a file name, and double-quoted strings are skipped. A caller scanning a
    root deck drops its title card first: line 1 is never read as a value.
    """
    sites: list[ValueSuffixSite] = []
    for card in cards:
        body = card.body
        if not body or body.isascii():
            continue
        if card.kind == "directive" and body.split(None, 1)[0].casefold() in INCLUDE_HEADS:
            continue
        for start, end in _unquoted_spans(body):
            for match in _SUFFIX_SITE_RE.finditer(body, start, end):
                number, suffix, tail = match.groups()
                sites.append(
                    ValueSuffixSite(
                        card=card,
                        offset=match.start(2),
                        line=card.line_at(match.start(2)),
                        number=number,
                        suffix=suffix,
                        tail=tail,
                    )
                )
    return sites


def fold_micro_suffix_cards(cards: list[SpiceCard]) -> tuple[ValueSuffixSite, ...]:
    """Spell every micro-sign scale suffix in ``cards`` as ``u``, in place.

    Returns the sites folded, whose ``token`` keeps the original spelling.
    Only the suffix character changes: comments, names, paths and every other
    character of the deck are left as they are.

    LTspice 24 and later write ``µ`` as UTF-8 (``C2 B5``); LTspice XVII
    decodes a deck as cp1252, reads those bytes as ``Âµ``, and drops the
    scale without a diagnostic, so ``23µ`` runs as 23. ``u`` means micro in
    every encoding and to every simulator.
    """
    folded = tuple(site for site in value_suffix_sites(cards) if site.micro)
    for site in folded:
        site.card.replace_span(site.offset, site.offset + 1, "u")
    return folded


#: What LTspice's netlister writes between the element letter it prefixes and
#: an instance name that does not start with that letter: ``R§Load``.
SECTION_SIGN = "§"


def strip_instance_section_signs(cards: list[SpiceCard]) -> dict[str, str]:
    """Drop ``§`` from every instance name that holds one, wherever it is named.

    ngspice's parser does not accept the character, so a deck LTspice exported
    names those instances without it for ngspice: on the instance card itself
    and in every other card that names the instance (``I(R§Load)`` in a
    ``.meas``). Nothing else changes: a ``§`` in a comment, a double-quoted
    string or an include path is part of a file name or text, not an instance.
    Returns each renamed instance's old name with its new one.
    """
    renamed = {
        card.instance_ref: card.instance_ref.replace(SECTION_SIGN, "")
        for card in cards
        if card.kind == "instance" and card.instance_ref and SECTION_SIGN in card.instance_ref
    }
    if not renamed:
        return {}
    by_name = {old.casefold(): new for old, new in renamed.items()}
    names = "|".join(re.escape(old) for old in sorted(renamed, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w{SECTION_SIGN}])(?:{names})(?![\w{SECTION_SIGN}])", re.I)
    for card in cards:
        body = card.body
        if SECTION_SIGN not in body:
            continue
        if card.kind == "directive" and body.split(None, 1)[0].casefold() in INCLUDE_HEADS:
            continue
        found = [
            m for start, end in _unquoted_spans(body) for m in pattern.finditer(body, start, end)
        ]
        for match in reversed(found):
            card.replace_span(match.start(), match.end(), by_name[match.group(0).casefold()])
    return renamed
