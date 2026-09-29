"""Structural (node-blind) comparison of two netlists.

What ``verify_circuit``'s ``structural_diff`` mode and the sidecar export's
``diff_vs_prior`` report: components added, removed or changed, and directives
added or removed. Both sides are netlists — an ``.asc`` is compared as its LTspice
export — read with the foundation lexer, so a ``+`` continuation belongs to the
card it continues and an inline comment is not part of it.

Two rules keep the delta quiet when nothing changed:

* Directives are compared parsed, not as text. Spacing, case, comma separators,
  the micro sign, and the order of ``.model``/``.param``/``.options`` assignments
  are spelling.
* LTspice's netlister adds lines to every export that say nothing about the
  circuit: ``.backanno``, the install's own ``.lib <install>/lib/cmp/standard.*``
  (at a path that differs per machine), and a parameterless default model such as
  ``.model NMOS NMOS`` per device class. The first two are left out of both sides;
  a default model is left out of the compared deck while the baseline declares no
  model of that name — once it does, the two disagree about what the model is.

Components are matched by reference the way the equivalence mode matches them
(``netlist_graph.canon_ref``) and compared on their model or value plus their
instance parameters. Nodes are left out on purpose: equivalence is the mode that
compares wiring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path

from ltspice_mcp.lib.format import fold_micro_sign
from ltspice_mcp.lib.netlist_graph import canon_ref
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    cards_from_path,
    lex,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import ModelCard, read_instance

# Directive cards whose assignments are a set: ``.param a=1 b=2`` and
# ``.param b=2 a=1`` define the same thing. (A ``.model`` is read through its
# typed view, whose parameters are a set by construction.)
_UNORDERED_ASSIGNMENTS = frozenset({".param", ".params", ".option", ".options", ".opt"})

# The parameterless default models LTspice's netlister declares for the stock
# device symbols (``.model NMOS NMOS`` and ``.model PMOS PMOS`` for a MOSFET, and
# likewise for the bipolar, JFET and diode symbols).
_DEFAULT_DEVICE_MODELS = frozenset({"nmos", "pmos", "npn", "pnp", "njf", "pjf", "d"})

# The install's own component library, which the netlister appends as
# ``.lib <install>/lib/cmp/standard.<kind>`` for every device class on the sheet.
# A bare ``standard.<kind>`` names the same file through the library search path.
_STANDARD_LIBRARY = re.compile(r"(?:.*[\\/])?cmp[\\/]standard\.\w+|standard\.\w+", re.IGNORECASE)

# Card kinds read as directives. ``end`` is not one: every deck has it, and a
# deck without it still means the same circuit.
_DIRECTIVE_KINDS = frozenset({"model", "param", "subckt", "ends", "meas", "directive"})


def comparable(text: str) -> str:
    """The equality form of a signature or directive: SPICE is case-insensitive,
    and the micro signs mean the same ``u`` suffix."""
    return fold_micro_sign(text).casefold()


def instance_signature(card: SpiceCard) -> str:
    """An instance card's model or value, then its parameters, sorted by name.

    Parameters are part of it because an attribute edit on a sheet (a
    SpiceLine's ``w=``, a diode's area) changes exactly those in the export; their
    order on the card carries no meaning. An unreadable card signs as
    ``<unparseable>``, as every other netlist read reports it.
    """
    inst = read_instance(card)
    if inst is None:
        return "<unparseable>"
    head = inst.display_value()
    parts = [head] if head else []
    if inst.model is not None and inst.value:
        parts.append(inst.value)  # what follows the model: a diode area, a switch state
    prefix = inst.ref[:1].upper()
    for key, value in sorted(inst.params.items(), key=lambda kv: kv[0].casefold()):
        pair = f"{key}={value}"
        if pair == head or (key.upper() == prefix and value == inst.value):
            continue  # already the head: a B-source's V=..., a keyed R=1k
        parts.append(pair)
    return " ".join(parts)


def _canonical_value(text: str) -> str:
    """A token's text with its insignificant whitespace removed."""
    if text.startswith("(") and text.endswith(")"):
        return "(" + " ".join(_canonical_words(text[1:-1])) + ")"
    if text.startswith("{") and text.endswith("}"):
        return "".join(text.split())
    return text


def _word(tok: Token) -> str:
    if tok.kind is TokenKind.KEY_VALUE:
        return f"{tok.key}={_canonical_value(tok.value or '')}"
    return _canonical_value(tok.text)


def _canonical_words(body: str) -> list[str]:
    """A card body as tokens: spacing, comma separators and the space around
    ``=`` or inside parentheses are spelling, not content."""
    tokens = takewhile(lambda t: t.kind is not TokenKind.COMMENT_TRAIL, tokenize_body(body))
    return [_word(tok) for tok in tokens]


@dataclass(frozen=True)
class Directive:
    """One directive card, parsed for comparison.

    ``text`` is the card as written — continuation lines joined, inline comment
    dropped, whitespace collapsed — and is what a delta reports. ``key`` decides
    equality. ``model`` is the name a ``.model`` card declares, case folded.
    """

    key: str
    text: str
    model: str | None = None

    @property
    def default_model(self) -> bool:
        """Whether this is one of the netlister's parameterless default models."""
        return (
            self.model in _DEFAULT_DEVICE_MODELS
            and self.key == f".model {self.model} {self.model}"
        )


def parse_directive(card: SpiceCard) -> Directive | None:
    """The comparable form of a directive card, or None for export boilerplate."""
    text = " ".join(card.body.split())
    head = text.split(" ", 1)[0].casefold()
    if head == ".backanno":
        return None
    if head == ".lib" and _STANDARD_LIBRARY.fullmatch(text[len(head) :].strip().strip("\"'")):
        return None
    try:
        if card.kind == "model":
            model = ModelCard.from_card(card)
            params = sorted(
                (f"{k}={_canonical_value(v)}" for k, v in model.params.items()), key=str.casefold
            )
            words = [head, model.name, model.type, *params]
            return Directive(comparable(" ".join(words)), text, model=model.name.casefold())
        words = _canonical_words(card.body)
    except SpiceLexError:
        return Directive(comparable(text), text)  # unbalanced: compare the spelling as written
    if head in _UNORDERED_ASSIGNMENTS:
        rest = words[1:]
        assignments = sorted((w for w in rest if "=" in w), key=str.casefold)
        words = words[:1] + [w for w in rest if "=" not in w] + assignments
    return Directive(comparable(" ".join(words)), text)


@dataclass(frozen=True)
class Deck:
    """A netlist as a structural diff reads it.

    ``components`` maps a reference as the equivalence mode matches it
    (``canon_ref``: LTspice's ``§`` instance marker dropped, case folded) to the
    reference as written and its signature.
    """

    components: dict[str, tuple[str, str]] = field(default_factory=dict)
    directives: list[Directive] = field(default_factory=list)


def read_deck(source: str | Path) -> Deck:
    """Read a netlist (a path, or its text) for a structural diff.

    Raises what reading the file raises; a caller decides what an unreadable
    deck means for its comparison.
    """
    cards = (cards_from_path(source) if isinstance(source, Path) else lex(source)).cards
    deck = Deck()
    for card in cards:
        if card.kind == "instance" and card.name:
            deck.components[canon_ref(card.name)] = (card.name, instance_signature(card))
        elif card.kind in _DIRECTIVE_KINDS:
            directive = parse_directive(card)
            if directive is not None:
                deck.directives.append(directive)
        elif card.kind == "control":
            # An ngspice .control script line: not a card, but part of the deck.
            line = " ".join(card.raw_lines[0].split())
            if line and not line.startswith("*"):
                deck.directives.append(Directive(comparable(line), line))
    return deck


def _by_key(directives: list[Directive]) -> dict[str, list[str]]:
    by_key: dict[str, list[str]] = {}
    for d in directives:
        by_key.setdefault(d.key, []).append(d.text)
    return by_key


def structural_delta(baseline: Deck, compared: Deck) -> dict[str, list]:
    """The added/removed/changed component and directive delta, baseline → compared."""
    a, b = baseline.components, compared.components
    declared = {d.model for d in baseline.directives if d.model is not None}
    da = _by_key(baseline.directives)
    db = _by_key(
        [d for d in compared.directives if not (d.default_model and d.model not in declared)]
    )
    return {
        "components_added": sorted(b[key][0] for key in b.keys() - a.keys()),
        "components_removed": sorted(a[key][0] for key in a.keys() - b.keys()),
        "components_changed": [
            {"reference": b[key][0], "before": a[key][1], "after": b[key][1]}
            for key in sorted(a.keys() & b.keys())
            if comparable(a[key][1]) != comparable(b[key][1])
        ],
        "directives_added": sorted(d for k in db.keys() - da.keys() for d in db[k]),
        "directives_removed": sorted(d for k in da.keys() - db.keys() for d in da[k]),
    }
