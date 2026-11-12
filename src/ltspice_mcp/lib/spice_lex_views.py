"""Typed views over ``SpiceCard`` instances.

Each view is an ephemeral wrapper built from one card via
``from_card()``. Views never own children — body cards live in the flat
``lex()`` list with their scope tuples. Setters mutate the view's
fields and flip the underlying card's ``dirty`` flag plus rewrite its
``raw_lines`` so ``emit()`` re-renders it correctly.

Mutation safety: views are short-lived. Holding two views over the
same card and mutating both is undefined behaviour; re-derive the view
after each mutation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.format import format_spice_value as _format_value
from ltspice_mcp.lib.format import is_scaled_number, parse_spice_value
from ltspice_mcp.lib.spice_lex import (
    MEAS_ANALYSIS_TOKENS,
    SpiceCard,
    SpiceLexError,
    SpiceLexErrorCategory,
    Token,
    TokenKind,
    copies_by_name,
    directive_head,
    strip_matching_quotes,
    tokenize_body,
)


def _require_kind(card: SpiceCard, expected: str) -> None:
    """Guard against from_card on the wrong card kind."""
    if card.kind != expected:
        raise ValueError(
            f"expected SpiceCard kind={expected!r}, got {card.kind!r} at line {card.line_start}"
        )


def _malformed(card: SpiceCard, message: str, suggestion: str = "") -> SpiceLexError:
    """Build a structured MALFORMED_CARD error for the given card."""
    return SpiceLexError(
        SpiceLexErrorCategory.MALFORMED_CARD,
        f"{message} at line {card.line_start}",
        body=card.body,
        suggestion=suggestion,
    )


def _pair(tok: Token) -> tuple[str, str]:
    """A ``KEY_VALUE`` token as its ``(key, value)``."""
    assert tok.key is not None
    assert tok.value is not None
    return tok.key, tok.value


@dataclass
class _Params:
    """A card's parameters as it gives them: every ``key=value`` in order, a
    key given twice listed twice. What a view's setters edit and its re-render
    writes, so no copy is lost to a dict that keeps one value per key; which
    copy a simulator reads is not recorded here.

    ``tokens`` holds each pair's token while every offset is good for an
    in-place edit, else None: after a re-render, or for parameters whose place
    in the card is not known.
    """

    pairs: list[tuple[str, str]] = field(default_factory=list)
    tokens: list[Token] | None = None

    @classmethod
    def of(cls, tokens: list[Token], *, in_place: bool = True) -> _Params:
        return cls([_pair(tok) for tok in tokens], list(tokens) if in_place else None)

    def as_dict(self) -> dict[str, str]:
        """One value per name as written: a name given twice in one spelling
        reads as its last copy."""
        return dict(self.pairs)

    def repeated(self) -> dict[str, list[str]]:
        """Each parameter given more than once, by its case-folded name, with
        every copy as ``key=value`` in order."""
        copies = copies_by_name((k, f"{k}={v}") for k, v in self.pairs)
        return {name: given for name, given in copies.items() if len(given) > 1}

    def rendered(self) -> list[str]:
        return [f"{k}={v}" for k, v in self.pairs]

    def set(self, card: SpiceCard, key: str, value: str, *, append_at: int | None = None) -> bool:
        """Set every copy of ``key``, matched without regard to case, or add it.

        True when the card was edited in place: each copy replaced, right to
        left so the offsets still to edit stay put, or a new key written at
        ``append_at``. False when the caller must render the card again from
        the pairs: a new key with nowhere to go, or a copy whose span cannot
        be replaced in place.
        """
        name = key.casefold()
        copies = [i for i, (k, _) in enumerate(self.pairs) if k.casefold() == name]
        if not copies:
            self.pairs.append((key, value))
            return append_at is not None and self._insert(card, append_at, key, value)
        for i in copies:
            self.pairs[i] = (self.pairs[i][0], value)
        try:
            for i in reversed(copies):
                self._replace(card, i, value)
        except ValueError:
            return False
        return True

    def remove(self, key: str) -> bool:
        """Remove every copy of ``key``; True when there was one, and the
        caller must render the card again."""
        name = key.casefold()
        kept = [pair for pair in self.pairs if pair[0].casefold() != name]
        if len(kept) == len(self.pairs):
            return False
        self.pairs = kept
        self.tokens = None
        return True

    def shift(self, edit_old_end: int, delta: int) -> None:
        """Move the tokens at or past ``edit_old_end`` by ``delta``."""
        if self.tokens is not None:
            self.tokens = _shift_tokens(self.tokens, edit_old_end, delta)

    def _insert(self, card: SpiceCard, at: int, key: str, value: str) -> bool:
        """Write the new last pair at body offset ``at``; False when it cannot go there."""
        if self.tokens is None:
            return False
        text = f" {key}={value}"
        try:
            card.replace_span(at, at, text)
        except ValueError:
            return False
        self.tokens.append(
            Token(
                kind=TokenKind.KEY_VALUE,
                text=text[1:],
                key=key,
                value=value,
                body_offset=at + 1,
                body_length=len(text) - 1,
            )
        )
        return True

    def _replace(self, card: SpiceCard, index: int, value: str) -> None:
        """Replace one pair's text in place; ``ValueError`` when it cannot be."""
        if self.tokens is None or self.tokens[index].body_offset < 0:
            raise ValueError("no in-place span for this parameter")
        tok = self.tokens[index]
        new_text = f"{tok.key}={value}"
        card.replace_span(tok.body_offset, tok.body_end, new_text)
        self.shift(tok.body_end, len(new_text) - tok.body_length)
        assert self.tokens is not None
        self.tokens[index] = replace(tok, text=new_text, value=value, body_length=len(new_text))


# ---------------------------------------------------------------------------
# .MODEL
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Typed view over a ``.MODEL NAME TYPE(...)`` card.

    ``params`` carries the raw value tokens (``"0.7"``, ``"{vto_n}"``,
    ``"100u"``); callers parse to numeric as needed. Keys are stored
    case-preserved; matching against them folds case at lookup.

    ``level`` is the card's ``LEVEL=`` as a number, None when it gives none
    or gives an expression. It is the one reading of a model's level: the
    card's own parameters, so a comment between its ``+`` lines that names
    another level is no part of it.
    """

    card: SpiceCard
    name: str
    type: str
    level: float | None
    _params: _Params = field(default_factory=_Params)
    #: The name's token, renamed in place while it is good.
    _name_token: Token | None = None
    #: Just inside the card's opening parenthesis, for a new parameter written
    #: in place: after the last one, or here when there is none.
    _in_parens: int | None = None

    @classmethod
    def from_card(cls, card: SpiceCard) -> ModelCard:
        _require_kind(card, "model")
        tokens = tokenize_body(card.body)
        # Expected shape: BARE(.MODEL) BARE(name) (BARE|QUOTED)(type)
        # PARENED(params) | KEY_VALUE...
        if len(tokens) < 3:
            raise _malformed(card, "malformed .MODEL card")
        # tokens[0] is ".MODEL" (BARE), tokens[1] is name, tokens[2] is type.
        name = strip_matching_quotes(tokens[1].text)
        type_ = strip_matching_quotes(tokens[2].text)
        # Params can come as a single PARENED group or as a sequence of
        # KEY_VALUE tokens after the type. A PARENED group's own tokens are
        # placed in the card by the group's offset, so they are edited where
        # they stand. The body holds no comment: each line's ends at the end
        # of the line, and the lexer leaves it out.
        found: list[Token] = []
        in_parens = None
        for tok in tokens[3:]:
            if tok.kind == TokenKind.PARENED:
                in_parens = tok.body_offset + 1
                inner = tokenize_body(tok.text[1:-1])
                own = [t for t in inner if t.kind == TokenKind.KEY_VALUE]
                found.extend(_shift_tokens(own, 0, in_parens))
            elif tok.kind == TokenKind.KEY_VALUE:
                found.append(tok)
        params = _Params.of(found)
        level = None
        for k, v in params.as_dict().items():
            if k.lower() == "level":
                try:
                    level = parse_spice_value(v)
                except ValueError:
                    level = None  # an expression: no level is known here
                break
        return cls(
            card=card,
            name=name,
            type=type_,
            level=level,
            _params=params,
            _name_token=tokens[1],
            _in_parens=in_parens,
        )

    @property
    def params(self) -> dict[str, str]:
        """The parameters by name as written, one value each; ``param_pairs``
        has every copy."""
        return self._params.as_dict()

    def param_pairs(self) -> list[tuple[str, str]]:
        """Every ``key=value`` on the card, in order, a key given twice listed twice."""
        return list(self._params.pairs)

    def repeated_params(self) -> dict[str, list[str]]:
        """Each parameter the card gives more than once (``_Params.repeated``)."""
        return self._params.repeated()

    def set_param(self, key: str, value: float | str) -> None:
        """Set every copy of a parameter, or add it, flipping the card dirty.
        In place where the card allows, a new one after the last inside the
        parentheses; otherwise the card is rendered again, every copy kept."""
        tokens = self._params.tokens
        append_at = tokens[-1].body_end if tokens else self._in_parens
        if not self._params.set(self.card, key, _format_value(value), append_at=append_at):
            self._canonical_rerender()

    def remove_param(self, key: str) -> None:
        """Remove every copy of a parameter (no-op if absent)."""
        if self._params.remove(key):
            self._canonical_rerender()

    def set_name(self, new_name: str) -> None:
        """Rename the model, in place where the card allows; otherwise the
        card is rendered again, every parameter copy kept."""
        self.name = new_name
        self.card.name = new_name
        tok = self._name_token
        if tok is not None:
            try:
                self.card.replace_span(tok.body_offset, tok.body_end, new_name)
            except ValueError:
                pass
            else:
                delta = len(new_name) - tok.body_length
                self._params.shift(tok.body_end, delta)
                if self._in_parens is not None:
                    self._in_parens += delta
                self._name_token = replace(tok, text=new_name, body_length=len(new_name))
                return
        self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        params_text = " ".join(self._params.rendered())
        body = f".MODEL {self.name} {self.type}({params_text})"
        self.card.replace_body(body)
        # The offsets are now stale, so later edits on this view render the
        # card again too; re-derive the view via from_card for in-place edits.
        self._params.tokens = None
        self._name_token = None
        self._in_parens = None


# ---------------------------------------------------------------------------
# .PARAM
# ---------------------------------------------------------------------------


@dataclass
class ParamCard:
    """Typed view over a single-param ``.PARAM NAME=VALUE`` card.

    Multi-param ``.PARAM a=1 b=2`` cards are not supported by this view
    (rare in practice); use raw ``SpiceCard`` access for those.
    """

    card: SpiceCard
    name: str
    value: str
    _kv_token: Token | None = None

    @classmethod
    def from_card(cls, card: SpiceCard) -> ParamCard:
        _require_kind(card, "param")
        tokens = tokenize_body(card.body)
        # tokens[0] = ".PARAM", tokens[1] = KEY_VALUE.
        if len(tokens) < 2 or tokens[1].kind != TokenKind.KEY_VALUE:
            raise _malformed(card, "malformed .PARAM card")
        kv = tokens[1]
        assert kv.key is not None
        assert kv.value is not None
        return cls(card=card, name=kv.key, value=kv.value, _kv_token=kv)

    def set_value(self, value: float | str) -> None:
        """Set the param value, flipping the card dirty."""
        new_value = _format_value(value)
        self.value = new_value
        tok = self._kv_token
        if tok is not None and tok.body_offset >= 0:
            new_text = f"{tok.key}={new_value}"
            try:
                self.card.replace_span(tok.body_offset, tok.body_end, new_text)
                self._kv_token = Token(
                    kind=TokenKind.KEY_VALUE,
                    text=new_text,
                    key=tok.key,
                    value=new_value,
                    body_offset=tok.body_offset,
                    body_length=len(new_text),
                )
                return
            except ValueError:
                pass
        self.card.replace_body(f".PARAM {self.name}={new_value}")
        self._kv_token = None


# ---------------------------------------------------------------------------
# Element instance lines
# ---------------------------------------------------------------------------


# Element kind classification. ``default_kind`` says how the last
# positional token should be interpreted **in the default syntactic form**:
# - "model": last positional is a model or subckt name (M/Q/J/X).
# - "value": last positional is a numeric value, expression, or gain
#   (R/C/L/V/I, plus E/G/F/H positional-gain forms).
# - "params_only": no positional value slot — value carried as KEY=VALUE
#   in params (B behavioural sources always; E/G when written with VALUE=
#   or POLY/TABLE — controlled by ``kv_means_params_only``).
# - "none": no value at all (K mutual-inductance carries refs as nodes).
#
# ``kv_means_params_only`` flips the kind to ``params_only`` when any
# KEY_VALUE token is present. Real SPICE flexes E/G/F/H between
# positional-gain (``E1 out 0 in 0 10``) and behavioural params-only
# (``E1 out 0 VALUE={...}``); the same prefix carries both shapes.
#
# ``min_nodes`` is a sanity check only.
ElementKind = Literal["model", "value", "params_only", "none"]


# ---------------------------------------------------------------------------
# V / I — the words after a source's nodes: its value and its own parameters
# ---------------------------------------------------------------------------

#: A source's own parameters by its letter, which both LTspice builds read
#: after its value whatever form the value takes (the ``source-parameters``
#: recordings): a voltage source's series resistance and parallel capacitance,
#: and either source's AC magnitude, written ``AC=2`` or ``AC 2``. A value edit
#: keeps them. Every other word after a source's nodes is its value, a PWL's
#: ``TIME_SCALE_FACTOR=`` before its points among them, and is replaced.
SOURCE_PARAMETERS: dict[str, frozenset[str]] = {
    "V": frozenset({"rser", "cpar", "ac"}),
    "I": frozenset({"ac"}),
}
#: Each own keyed parameter as a message names it.
SOURCE_PARAMETER_NAMES = {"rser": "Rser=", "cpar": "Cpar=", "ac": "AC="}
#: The flag after a current source's value that makes it a load, which both
#: builds read there (``deck/source_parameters``).
LOAD_FLAG = "load"
#: The keyword of a source that plays a .wav file, which sets its level in
#: place of a value (``deck/wavefile_nodes``).
WAVEFILE = "wavefile"


@dataclass(frozen=True)
class SourceText:
    """The words after a V or I source's nodes: its value, its own parameters
    in order, each as written, and the ``wavefile=`` word of a source that
    plays a .wav file."""

    value: str
    own: tuple[str, ...]
    played: str | None = None

    @property
    def text(self) -> str:
        """The value then the own parameters, one space apart."""
        return " ".join(word for word in (self.value, *self.own) if word)


def text_without(text: str, spans: list[tuple[int, int]], end: int | None = None) -> str:
    """``text`` up to ``end`` with each of ``spans`` (start, stop) cut out, in
    order, the runs left kept as written and joined one space apart: a
    value with some of its words taken out."""
    return " ".join(text[start:stop] for start, stop in _runs_without(text, spans, end))


def _runs_without(
    text: str, spans: list[tuple[int, int]], end: int | None = None
) -> list[tuple[int, int]]:
    """The (start, stop) of each run of ``text`` up to ``end`` that ``spans``
    leave, in order, without its surrounding whitespace; empty runs left out."""
    runs, cursor = [], 0
    for start, stop in [*spans, (len(text) if end is None else end, len(text))]:
        piece = text[cursor:start]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            runs.append((cursor + lead, cursor + len(piece.rstrip())))
        cursor = stop
    return runs


def source_word_key(word: str) -> str:
    """The name a source's own parameter goes by: its key (``AC=2`` and
    ``AC 2`` alike), or the flag itself."""
    return word.split("=", 1)[0].split(None, 1)[0].strip().casefold()


def _is_amount(token: Token) -> bool:
    """A word that can be an AC magnitude or phase: a number or a braced expression."""
    return token.kind == TokenKind.BRACED or (
        token.kind == TokenKind.BARE and is_scaled_number(token.text)
    )


@dataclass(frozen=True)
class _SourceReading:
    """``read_source_text``'s answer with where it found each part in the
    text it read: the runs of the value, and the tokens of the keyed own
    parameters (``Rser=1k``, ``AC=2``)."""

    text: SourceText
    value_runs: list[tuple[int, int]]
    keyed: list[Token]


def _read_source(text: str, prefix: str) -> _SourceReading:
    """``read_source_text``, with where each part is in ``text``."""
    letter = prefix[:1].upper()
    own_keys = SOURCE_PARAMETERS.get(letter, frozenset())
    tokens = []
    end = len(text)
    for tok in tokenize_body(text):
        if tok.kind == TokenKind.COMMENT_TRAIL:
            end = tok.body_offset
            break
        tokens.append(tok)
    own: list[tuple[int, int]] = []
    keyed: list[Token] = []
    played = None
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        key = (tok.key or "").casefold() if tok.kind == TokenKind.KEY_VALUE else ""
        bare = tok.text.casefold() if tok.kind == TokenKind.BARE else ""
        if key in own_keys or (letter == "I" and bare == LOAD_FLAG):
            own.append((tok.body_offset, tok.body_end))
            if key:
                keyed.append(tok)
        elif bare == "ac" and "ac" in own_keys:
            # AC, its magnitude, and a phase where one follows.
            amounts = 0
            while amounts < 2 and index + 1 < len(tokens) and _is_amount(tokens[index + 1]):
                index += 1
                amounts += 1
            own.append((tok.body_offset, tokens[index].body_end))
        elif key == WAVEFILE:
            played = text[tok.body_offset : tok.body_end]
        index += 1
    runs = _runs_without(text, own, end)
    return _SourceReading(
        SourceText(
            value=" ".join(text[start:stop] for start, stop in runs),
            own=tuple(" ".join(text[start:stop].split()) for start, stop in own),
            played=played,
        ),
        runs,
        keyed,
    )


def read_source_text(text: str, prefix: str) -> SourceText:
    """``text``, the words after a source's nodes (on a card, or in a sheet
    part's Value), read as its value and its own parameters
    (``SOURCE_PARAMETERS``, ``AC`` with its magnitude and any phase, and a
    current source's ``load``): the one rule every reader of a source's value
    reads it by. ``InstanceLine`` reads a V or I card through it, and every
    edit of a source's value keeps the own parameters by it.

    Raises ``SpiceLexError`` for text that does not split into words.
    """
    return _read_source(text, prefix).text


def _source_value_start(tokens: list[Token]) -> int | None:
    """Where the words after a V or I card's two nodes start in its body, when
    the source rule (``read_source_text``) reads them: the card's first two
    words after its reference are nodes, whether or not a word follows them
    (``V1 a 0`` has no value, as a value edit reads it). None for every other
    card, and for a source that plays a .wav file, whose words are all nodes
    (``_DEFINING_KEYS``).
    """
    if (
        not tokens
        or tokens[0].kind != TokenKind.BARE
        or tokens[0].text[:1].upper() not in SOURCE_PARAMETERS
    ):
        return None
    words = [t for t in tokens[1:] if t.kind != TokenKind.COMMENT_TRAIL]
    if len(words) < 2 or any(t.kind == TokenKind.KEY_VALUE for t in words[:2]):
        return None
    if any((t.key or "").casefold() == WAVEFILE for t in words):
        return None
    return words[1].body_end


@dataclass(frozen=True)
class ElementSpec:
    default_kind: ElementKind
    min_nodes: int
    kv_means_params_only: bool = False
    # The words between the element's ``min_nodes`` nodes and its model name
    # the source whose current controls it (``W1 a 0 Vsense sw``).
    controlled: bool = False

    def kind_for(self, has_kv: bool) -> ElementKind:
        if has_kv and self.kv_means_params_only:
            return "params_only"
        return self.default_kind


ELEMENT_SPECS: dict[str, ElementSpec] = {
    # Model-bearing devices.
    "M": ElementSpec(default_kind="model", min_nodes=3),
    "Q": ElementSpec(default_kind="model", min_nodes=3),
    "J": ElementSpec(default_kind="model", min_nodes=3),
    "X": ElementSpec(default_kind="model", min_nodes=1),
    # The node counts of these are what both LTspice builds ran
    # (element-terminals): a MESFET's drain, gate and source, then its model
    # and an area factor (deck/element_z); a lossy line's two ports and an RC
    # line's two ends and the node its capacitance goes to, each before its
    # model (deck/element_o, deck/element_u); a current-controlled switch's two
    # nodes, then the source whose current sets it, then its model
    # (deck/element_w).
    "Z": ElementSpec(default_kind="model", min_nodes=3),
    "O": ElementSpec(default_kind="model", min_nodes=4),
    "U": ElementSpec(default_kind="model", min_nodes=3),
    "W": ElementSpec(default_kind="model", min_nodes=2, controlled=True),
    # A lossless line's two ports, and LTspice 26's frequency response analyzer
    # between two nodes: nodes, then parameters only (deck/element_t,
    # deck/element_fra; LTspice XVII has no analyzer).
    "T": ElementSpec(default_kind="params_only", min_nodes=4),
    "@": ElementSpec(default_kind="params_only", min_nodes=2),
    # Passives and independent sources: positional value, never params-only.
    "R": ElementSpec(default_kind="value", min_nodes=2),
    "C": ElementSpec(default_kind="value", min_nodes=2),
    "L": ElementSpec(default_kind="value", min_nodes=2),
    "V": ElementSpec(default_kind="value", min_nodes=2),
    "I": ElementSpec(default_kind="value", min_nodes=2),
    # Behavioural source — always params-only.
    "B": ElementSpec(default_kind="params_only", min_nodes=2),
    # Controlled sources — positional gain by default; params-only when
    # written with ``VALUE=``, ``POLY``, etc. (any KEY_VALUE present).
    "E": ElementSpec(default_kind="value", min_nodes=4, kv_means_params_only=True),
    "G": ElementSpec(default_kind="value", min_nodes=4, kv_means_params_only=True),
    "F": ElementSpec(default_kind="value", min_nodes=2, kv_means_params_only=True),
    "H": ElementSpec(default_kind="value", min_nodes=2, kv_means_params_only=True),
    # Mutual inductance: refs only.
    "K": ElementSpec(default_kind="none", min_nodes=0),
}

_DEFAULT_ELEMENT_SPEC = ElementSpec(default_kind="model", min_nodes=0)

# EXACT terminal count per element class, for node-connectivity editing
# (set_nodes) and value/node splitting. Only classes whose node count is GENUINELY
# fixed are listed: for these the FIRST N positional tokens are always the nodes,
# so the node span can be rewritten (or the value tail split off, e.g. a source's
# ``PULSE(...)``) without parsing the rest of the line. Variable/ambiguous classes
# are deliberately ABSENT so set_nodes refuses them rather than corrupt a valid
# form: BJT/JFET (Q/J) take an optional 4th substrate node; controlled sources
# (E/G) have POLY/TABLE forms with variable control-node arity; MOSFET (M, 3-vs-4
# bulk + trailing area/off), subcircuit (X), and mutual inductance (K) are all
# variable too. (D stays: a diode is always 2 nodes, with any area/off trailing.)
_EXACT_NODE_COUNT: dict[str, int] = {
    "R": 2,
    "C": 2,
    "L": 2,
    "V": 2,
    "I": 2,
    "D": 2,
    "F": 2,
    "H": 2,
}

# The keywords that carry a two-terminal element's value in place of its
# positional value, so every positional word is a node: a capacitor defined by
# its charge or an inductor by its flux (``C1 b 0 Q=100n*x``,
# ``L1 c 0 Flux=1m*x``), and a source that plays a .wav file
# (``V1 a b wavefile="in.wav"``), whose second node both LTspice builds wire
# as written, quoted or not (wavefile-source-nodes).
_DEFINING_KEYS: dict[str, frozenset[str]] = {
    "C": frozenset({"q"}),
    "L": frozenset({"flux"}),
    "V": frozenset({"wavefile"}),
    "I": frozenset({"wavefile"}),
}

# A switch (S/W) carries an optional ON/OFF state token AFTER its model name.
# It is never a model name, so it is peeled into the value tail before model
# identification — otherwise the last-positional heuristic would mistake the
# state for the model and clobber the real model on a set_model edit.
_SWITCH_STATES = frozenset({"on", "off"})

# Devices whose model may be followed by an area factor and an OFF
# (initial-condition) keyword: ``Q2 c b e QN 8``, ``J1 d g s JN 2 off``,
# ``M1 d g s b NCH off``, ``Z2 d g s NMF 2`` (a MESFET of twice the area,
# deck/element_z).
_DEVICE_TAIL_PREFIXES = frozenset({"Q", "J", "M", "Z"})


def _is_device_tail_value(token: Token) -> bool:
    """A positional that can only be a device's trailing area factor or state.

    A number, a ``{...}`` or ngspice ``'...'`` expression, or ON/OFF — never a
    model name.
    """
    if token.kind == TokenKind.BRACED:
        return True
    if token.kind == TokenKind.QUOTED:
        return token.text.startswith("'")
    if token.kind != TokenKind.BARE:
        return False
    # Strictly a number: a model name that starts with digits (``2N2222``)
    # must not read as an area factor.
    return token.text.casefold() in _SWITCH_STATES or is_scaled_number(token.text)


def _exact_node_span(positional: list[Token], exact: int | None) -> tuple[int, int] | None:
    """Body span of the FIRST ``exact`` positional tokens — the editable node
    region of a genuinely fixed-arity element. ``None`` when the arity is unknown,
    there are too few tokens, or an offset is synthesized. Computed independently
    of the per-kind ``nodes`` parse so it stays correct for a model-kind element
    in the table (e.g. a diode whose trailing area token would otherwise be
    miscounted as a node)."""
    if exact is None or exact < 1 or len(positional) < exact:
        return None
    head = positional[:exact]
    if any(t.body_offset < 0 for t in head):
        return None
    return (head[0].body_offset, head[-1].body_end)


def _value_from_span(card: SpiceCard, tokens: list[Token]) -> str:
    """Reconstruct the value text spanning ``tokens`` from the card body.

    Prefers the original body span (so spacing inside a multi-token spec like
    ``PULSE(...)`` survives a round-trip) and falls back to a space-join when a
    token is synthesized (no body offset)."""
    if all(t.body_offset >= 0 for t in tokens):
        return card.body[tokens[0].body_offset : tokens[-1].body_end].strip()
    return " ".join(t.text for t in tokens)


@dataclass
class InstanceLine:
    """Typed view over an element instance card.

    Field semantics by element kind:

    - ``M`` / ``Q`` / ``J`` / ``Z`` / ``X``: ``model`` carries the model or
      subckt name; ``value`` carries a trailing area factor or OFF
      (``Q2 c b e QN 8``), else None.
    - ``O`` / ``U`` / ``D`` / ``S``: ``model`` carries the model name.
    - ``W``: ``model`` carries the switch model; the source whose current
      sets the switch, written between its two nodes and its model, is
      neither a node nor the model.
    - ``R`` / ``C`` / ``L``: ``value`` carries the passive value (e.g.
      ``"1k"``), ``model`` is None. A value carried by a keyword (``R=``, a
      capacitor's ``Q=``, an inductor's ``Flux=``) leaves every positional
      a node.
    - ``V`` / ``I``: the words after the two nodes are read by the source
      rule (``read_source_text``) into ``source``: ``value`` is its value
      (``"PULSE(0 1 0 1n 1n 1u 2u)"``, ``"PWL TIME_SCALE_FACTOR=2 (0 4 1
      4)"``), None when it gives none; its own keyed parameters (``Rser=``,
      ``Cpar=``, ``AC=``) are ``params``, and its own words without ``=``
      (``AC 2``, a current source's ``load``) are ``own_words``. A source
      that plays a .wav file (``wavefile=``) has every positional a node and
      no value.
    - ``B`` / ``E`` / ``G``: both are None — the value lives in
      ``params["V"]`` or ``params["I"]``.
    - ``T`` / ``@``: both are None — every positional is a node and the
      rest are parameters.
    - ``F`` / ``H``: ``value`` carries the controlling source name and gain.
    - ``K``: ``model`` carries the inductor pair or coupling
      coefficient — depends on syntax.

    ``nodes`` are positional tokens between the ref and the
    model/value token.
    """

    card: SpiceCard
    ref: str
    nodes: list[str]
    model: str | None
    value: str | None
    # The words between a ``W`` switch's nodes and its model, which name the
    # source whose current sets it; written back there when the card is
    # re-rendered. None for every other element.
    control: str | None = None
    _model_token: Token | None = None
    _value_param_key: str | None = None
    # A subckt call's ``params:`` keyword as written, put back in front of the
    # overrides when the card is re-rendered.
    _params_marker: str | None = None
    # Body span (start, end) covering exactly the positional node tokens, so
    # set_nodes can rewrite connectivity without disturbing the value/params
    # tail. None when there are no nodes or their offsets are synthesized.
    _node_span: tuple[int, int] | None = None
    _params: _Params = field(default_factory=_Params)
    # A V or I source's words after its nodes as its value and its own
    # parameters (``read_source_text``); None for every other card, and for a
    # source the rule does not read (one that plays a .wav file).
    source: SourceText | None = None
    # A source's words after its nodes with its keyed own parameters cut out,
    # in card order: what display shows of it.
    _source_shown: str = ""

    @classmethod
    def from_card(cls, card: SpiceCard) -> InstanceLine:
        _require_kind(card, "instance")
        tokens = tokenize_body(card.body)
        if not tokens:
            raise _malformed(card, "empty instance card")
        if tokens[0].kind != TokenKind.BARE:
            raise _malformed(card, "instance card does not start with a bare ref")
        start = _source_value_start(tokens)
        if start is not None:
            return cls._from_source(card, tokens, start)
        ref = tokens[0].text
        prefix = ref[:1].upper()

        first_kv = None
        for idx, tok in enumerate(tokens[1:], start=1):
            if tok.kind == TokenKind.KEY_VALUE:
                first_kv = idx
                break

        positional = tokens[1:first_kv] if first_kv is not None else tokens[1:]
        positional = [t for t in positional if t.kind != TokenKind.COMMENT_TRAIL]

        spec = ELEMENT_SPECS.get(prefix, _DEFAULT_ELEMENT_SPEC)
        kind = spec.kind_for(has_kv=first_kv is not None)
        # Exact terminal count for genuinely fixed-arity classes; drives both the
        # value/node split below and the node-edit span. ``None`` for everything
        # else (node editing then refuses it).
        exact = _EXACT_NODE_COUNT.get(prefix)
        model: str | None = None
        value: str | None = None
        model_token: Token | None = None
        params_marker: str | None = None
        control: str | None = None
        nodes_tokens: list[Token]

        if not positional:
            nodes_tokens = []
        elif kind == "params_only":
            # Behavioural / VALUE= / POLY / TABLE form: every positional
            # is a node; the value lives in a KEY_VALUE token.
            nodes_tokens = positional
        elif kind == "none":
            # K mutual-inductance — positional tokens are refs, not nodes.
            nodes_tokens = positional
        elif kind == "value":
            # Value-bearing element. The value may be MULTI-token: a source
            # function spec like ``PULSE(0 5 ...)`` lexes as ``PULSE`` + ``(...)``,
            # and a ``V1 a 0 DC 5 AC 1`` form is several tokens — so split on the
            # element's EXACT terminal count, not "last token is the value".
            # Reconstruct the value text from the body span so original spacing
            # (``PULSE(...)`` with no gap) survives.
            if exact is not None and len(positional) > exact:
                nodes_tokens = positional[:exact]
                value_tokens = positional[exact:]
                model_token = value_tokens[-1]
                value = _value_from_span(card, value_tokens)
            else:
                model_token = positional[-1]
                value = model_token.text.strip('"')
                nodes_tokens = positional[:-1]
        else:
            # "model": the model/subckt name, optionally followed by a trailing
            # value the last-positional heuristic would mistake for the model:
            #  - a diode area factor (``D1 a k 1N4148 2`` — the model is the
            #    first positional after its fixed 2 nodes, the 2 is the area);
            #  - a switch ON/OFF state (``S1 ... MYSW ON``);
            #  - a BJT/JFET/MOSFET area factor or OFF (``Q2 c b e QN 8``,
            #    ``M1 d g s b NCH off``), peeled only while the tokens left
            #    still hold the minimum nodes plus a model, so ``Q1 c b e 555``
            #    keeps a numerically-named model.
            # All of them follow the model and are preserved as the value tail.
            # A subckt call's ``params:`` keyword introduces its overrides; it
            # is neither the subckt name nor a value, so it is dropped here.
            tail: list[Token] = []
            pos = positional
            # ON/OFF is a state ONLY for switches (S/W). For any other element a
            # trailing ON/OFF is the model/subckt name itself (e.g. ``D1 a k ON``),
            # so peeling it there would corrupt the parse.
            if prefix in ("S", "W") and pos[-1].text.strip('"').lower() in _SWITCH_STATES:
                tail = [pos[-1]]
                pos = pos[:-1]
            elif prefix == "X" and pos[-1].text.casefold() == "params:":
                params_marker = pos[-1].text
                pos = pos[:-1]
            elif prefix in _DEVICE_TAIL_PREFIXES:
                while len(pos) > spec.min_nodes + 1 and _is_device_tail_value(pos[-1]):
                    tail.insert(0, pos[-1])
                    pos = pos[:-1]
            lead = spec.min_nodes
            if spec.controlled and len(pos) > lead + 1:
                # A W switch: its nodes, the source that sets it, its model.
                nodes_tokens = pos[:lead]
                control = _value_from_span(card, pos[lead:-1])
                model_token = pos[-1]
                model = model_token.text.strip('"')
            elif pos:
                # Fixed-arity (a diode) splits nodes | model | trailing area at the
                # known node count; everything else takes the last positional as
                # the model. (``pos`` can be empty only for a malformed bare
                # switch state like ``S1 ON``.)
                cut = exact if (exact is not None and len(pos) > exact) else len(pos) - 1
                nodes_tokens = pos[:cut]
                model_token = pos[cut]
                model = model_token.text.strip('"')
                tail = [*pos[cut + 1 :], *tail]
            else:
                nodes_tokens = []
            if tail:
                value = _value_from_span(card, tail)

        nodes = [t.text for t in nodes_tokens]

        # A comment trail is the lexer's last token, so every KEY_VALUE token
        # is a parameter, the first of them at first_kv.
        kv = _Params.of([tok for tok in tokens if tok.kind == TokenKind.KEY_VALUE])
        params = kv.as_dict()

        value_param_key: str | None = None
        if kind == "value" and prefix in ("R", "C", "L"):
            value_param_key = next(
                (key for key in params if key.upper() == prefix),
                None,
            )
        if value_param_key is not None:
            # Keyed primary-value form: ``R1 a b R=1k``. All
            # positionals are nodes; the value lives in the matching
            # KEY_VALUE token and must be edited through set_param.
            nodes = [t.text for t in positional]
            value = params[value_param_key]
        elif any(key.casefold() in _DEFINING_KEYS.get(prefix, frozenset()) for key in params):
            # A capacitor defined by its charge, an inductor by its flux
            # (``C1 b 0 Q=100n*x``), a source that plays a .wav file: both
            # builds run each as a part between its two nodes
            # (deck/keyword_values, deck/wavefile_nodes). Every positional is
            # a node and the card has no value.
            nodes = [t.text for t in positional]
            value = None
            model_token = None

        # Node-edit span (set_nodes): the first N positional tokens for a
        # fixed-arity element. Computed from the exact count, not the per-kind
        # nodes parse, so it stays correct for a model-kind table entry (a diode
        # with a trailing area token).
        node_span = _exact_node_span(positional, exact)

        return cls(
            card=card,
            ref=ref,
            nodes=nodes,
            model=model,
            value=value,
            _model_token=model_token,
            _value_param_key=value_param_key,
            _params_marker=params_marker,
            control=control,
            _node_span=node_span,
            _params=kv,
        )

    @classmethod
    def _from_source(cls, card: SpiceCard, tokens: list[Token], start: int) -> InstanceLine:
        """A V or I card read by the source rule, whose words after its two
        nodes start at body offset ``start`` (``_source_value_start``)."""
        nodes = [t for t in tokens[1:] if t.kind != TokenKind.COMMENT_TRAIL][:2]
        reading = _read_source(card.body[start:], tokens[0].text)
        value_token = None
        if len(reading.value_runs) == 1:
            # One run of words, which ``set_value`` replaces in place.
            first, last = (start + offset for offset in reading.value_runs[0])
            value_token = Token(
                kind=TokenKind.BARE,
                text=card.body[first:last],
                body_offset=first,
                body_length=last - first,
            )
        keyed = [replace(tok, body_offset=tok.body_offset + start) for tok in reading.keyed]
        tail = card.body[start:]
        cut = [(tok.body_offset, tok.body_end) for tok in reading.keyed]
        end = next(
            (t.body_offset - start for t in tokens if t.kind == TokenKind.COMMENT_TRAIL), None
        )
        return cls(
            card=card,
            ref=tokens[0].text,
            nodes=[t.text for t in nodes],
            model=None,
            value=reading.text.value or None,
            _model_token=value_token,
            _node_span=_exact_node_span(nodes, 2),
            _params=_Params.of(keyed),
            source=reading.text,
            _source_shown=text_without(tail, cut, end),
        )

    @property
    def own_words(self) -> tuple[str, ...]:
        """A V or I source's own parameters written without ``=`` (``AC 2``, a
        current source's ``load``), in card order; its keyed ones are in
        ``params``. Empty for every other card."""
        return tuple(word for word in self.source.own if "=" not in word) if self.source else ()

    def set_model(self, name: str) -> None:
        """Replace the model/subckt-name token."""
        self.model = name
        self._update_slot_in_place(name)

    def set_value(self, value: float | str) -> None:
        """Replace the value token (for R/C/L/V/I)."""
        new = _format_value(value)
        self.value = new
        if self._value_param_key is not None:
            self.set_param(self._value_param_key, new)
            return
        self._update_slot_in_place(new)

    def set_nodes(self, new_nodes: list[str]) -> None:
        """Rewrite the positional node list (connectivity edit).

        Replaces only the leading node-token span, leaving the value/model/params
        tail byte-for-byte — a canonical re-render would risk mangling multi-token
        source specs (e.g. ``PULSE(...)``). Supported only for genuinely fixed-arity
        classes (see ``_EXACT_NODE_COUNT``); a rewire must keep the terminal count.
        Variable/ambiguous classes raise — BJT/JFET (optional substrate node),
        controlled sources (POLY/TABLE forms), MOSFET (bulk + area/off), subcircuit,
        and mutual inductance — edit those cards directly.
        """
        prefix = self.ref[:1].upper()
        exact = _EXACT_NODE_COUNT.get(prefix)
        if exact is None or self._node_span is None:
            raise ValueError(
                f"node editing is not supported for {self.ref} (variable or "
                "ambiguous terminal count) — edit the card directly"
            )
        if len(new_nodes) != exact:
            raise ValueError(
                f"{self.ref} has {exact} node(s); got {len(new_nodes)} "
                "(a rewire keeps the terminal count)"
            )
        if any((not n) or n.split() != [n] for n in new_nodes):
            raise ValueError("node names must be non-empty single tokens")
        start, end = self._node_span
        self.nodes = list(new_nodes)
        self._node_span = None
        try:
            self.card.replace_span(start, end, " ".join(new_nodes))
        except ValueError:
            # The node span crosses a continuation-line boundary, so it can't be
            # patched in place; re-render canonically instead of propagating —
            # matching set_model/set_param's fallback.
            self._canonical_rerender()
            return
        # Offsets past the edit are now stale; drop cached tokens so any
        # subsequent edit on this view re-renders canonically instead of
        # writing at a wrong offset.
        self._model_token = None
        self._params.tokens = None

    @property
    def params(self) -> dict[str, str]:
        """The parameters by name as written, one value each: a key the card
        gives twice in one spelling reads as its last copy. ``param_pairs``
        has every copy."""
        return self._params.as_dict()

    def param_pairs(self) -> list[tuple[str, str]]:
        """Every ``key=value`` on the card as it now reads, in order, a key the
        card names twice listed twice. ``params`` keeps one value per key, so
        it cannot say that a card gives a parameter more than once."""
        return list(self._params.pairs)

    def repeated_params(self) -> dict[str, list[str]]:
        """Each parameter the card gives more than once (``_Params.repeated``)."""
        return self._params.repeated()

    def get_param(self, key: str) -> str | None:
        """Case-insensitive parameter read, matching the CI semantics of
        ``set_param``/``remove_param``. ``params`` preserves source casing
        (ngspice decks conventionally write lowercase ``w=``/``l=``), so a
        plain dict lookup silently misses — readers must come through here.

        A parameter the card gives more than once reads as its value when
        every copy agrees, and copies that differ are refused: which one a
        simulator reads is not recorded here.
        """
        name = key.casefold()
        copies = [(k, v) for k, v in self._params.pairs if k.casefold() == name]
        if len({v for _k, v in copies}) > 1:
            given = ", ".join(f"{k}={v}" for k, v in copies)
            raise NetlistError(
                f"{self.ref} gives {key} {len(copies)} times with different values ({given}), "
                "and which one the simulator reads is not known. Keep one."
            )
        return copies[0][1] if copies else None

    def set_param(self, key: str, value: float | str) -> None:
        """Set a parameter, or add it, flipping the card dirty.

        Every copy the card gives of ``key`` is set, matched without regard to
        case: setting one left the others at their old value, and which copy a
        simulator reads is not recorded here. In place where the card allows;
        otherwise, and for a new key, the card is rendered again with every
        copy kept. A source's own parameter may come before its value
        (``V1 a 0 AC=1 PULSE(...)``), so a later value edit on this view then
        renders the card again rather than write at an offset the edit moved.

        A V or I source has only its own parameters (``SOURCE_PARAMETERS``):
        any other ``key=`` after its nodes is part of its value, so setting one
        is refused, naming the fix.
        """
        if self.source is not None:
            self._refuse_unowned(key)
        value_token = self._model_token
        before_value = value_token is not None and any(
            tok.body_offset < value_token.body_offset for tok in self._params.tokens or ()
        )
        if not self._params.set(self.card, key, _format_value(value)):
            self._canonical_rerender()
        elif before_value:
            self._model_token = None

    def _refuse_unowned(self, key: str) -> None:
        """Refuse ``key`` on a source that does not own it."""
        letter = self.ref[:1].upper()
        if key.casefold() in SOURCE_PARAMETERS[letter]:
            return
        source = "an I source" if letter == "I" else "a V source"
        own = ", ".join(SOURCE_PARAMETER_NAMES[name] for name in sorted(SOURCE_PARAMETERS[letter]))
        written = any(
            (tok.key or "").casefold() == key.casefold()
            for tok in tokenize_body(self.value or "")
            if tok.kind == TokenKind.KEY_VALUE
        )
        if written:
            raise NetlistError(
                f"{self.ref}: {key}= is part of its value ({self.value}), not a parameter "
                f"of {source}, whose own are {own}. Edit the value instead."
            )
        raise NetlistError(
            f"{self.ref}: {key}= is no parameter of {source}, whose own are {own}; any "
            f"other key=value after its nodes is read as part of its value. Edit the "
            f"value to change it."
        )

    def remove_param(self, key: str) -> None:
        """Remove every copy of a parameter (no-op if absent)."""
        if self._params.remove(key):
            self._canonical_rerender()

    def _update_slot_in_place(self, new_text: str) -> None:
        """Try replace_span on the model/value token; fall back to canonical.

        On in-place success, shifts every cached param token whose
        body_offset sat past the model-token edit by the length delta.
        """
        tok = self._model_token
        if tok is not None and tok.body_offset >= 0:
            old_body_end = tok.body_end
            delta = len(new_text) - tok.body_length
            try:
                self.card.replace_span(tok.body_offset, old_body_end, new_text)
                self._model_token = Token(
                    kind=tok.kind,
                    text=new_text,
                    body_offset=tok.body_offset,
                    body_length=len(new_text),
                )
                self._params.shift(old_body_end, delta)
                return
            except ValueError:
                pass
        self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        parts = [self.ref, *self.nodes]
        if self.control is not None:
            parts.append(self.control)
        if self.model is not None:
            parts.append(self.model)
            # A model-kind element may carry a trailing value (diode area, switch
            # ON/OFF state) after the model name; preserve it on re-render.
            if self.value is not None:
                parts.append(self.value)
        elif self.value is not None and self._value_param_key is None:
            parts.append(self.value)
        parts.extend(self.own_words)
        if self._params_marker is not None:
            parts.append(self._params_marker)
        parts.extend(self._params.rendered())
        body = " ".join(parts)
        self.card.replace_body(body)
        self._model_token = None
        self._params.tokens = None

    def display_value(self) -> str:
        """Single-string projection for display: model name for M/Q/J/X,
        the controlling source and model for a W switch, value field for
        R/C/L/E/F/G/H, a V or I source's value then its own words without
        ``=`` (``1m load``), KV pair for B-sources, empty for malformed
        bodies."""
        if self.model is not None:
            return self.model if self.control is None else f"{self.control} {self.model}"
        if self.source is not None:
            # Its value and the own parameters written without ``=``, in card
            # order (``DC 0 AC 1 SIN(0 1 1k)``).
            return self._source_shown
        if self.value is not None:
            return self.value
        for key in ("V", "I", "v", "i"):
            if key in self.params:
                return f"{key}={self.params[key]}"
        if self.params:
            k0, v0 = next(iter(self.params.items()))
            return f"{k0}={v0}"
        return ""


def body_has_stray_kv_remnant(body: str) -> bool:
    """True iff the body's KV parser left orphan tokens after the first
    KEY=VALUE. Callers surface ``<unparseable>`` rather than render a
    truncated value. Never for a V or I source the source rule reads, whose
    words after its nodes are each its value's or its own parameter's."""
    try:
        toks = tokenize_body(body)
    except Exception:
        return False
    if _source_value_start(toks) is not None:
        # The source rule reads every word after a source's nodes as its value
        # or one of its own parameters, a PWL's ``TIME_SCALE_FACTOR=2`` before
        # its points among them (deck/source_parameters).
        return False
    seen_kv = False
    for tok in toks:
        if tok.kind == TokenKind.KEY_VALUE:
            seen_kv = True
            continue
        if seen_kv and tok.kind in (TokenKind.PARENED, TokenKind.BARE):
            return True
    return False


def read_instance(card: SpiceCard) -> InstanceLine | None:
    """The typed view of an instance card, or None when it cannot be read.

    A body that lexed but carries a broken ``k=v`` remnant is unreadable too,
    since its view would describe a truncated card. The one rule behind every
    ``<unparseable>`` a netlist read reports.
    """
    # A remnant follows a KEY=VALUE, so a body with no '=' cannot carry one.
    if "=" in card.body and body_has_stray_kv_remnant(card.body):
        return None
    try:
        return InstanceLine.from_card(card)
    except Exception:  # any view fault is an unreadable card
        return None


def instances_by_ref(
    cards: Iterable[SpiceCard], *, scope: tuple[str, ...] | None = None
) -> dict[str, SpiceCard]:
    """Lowercased ``ref`` → instance-card lookup. Skips cards with no ref.

    ``scope`` keeps only the cards directly in that scope: ``()`` is the top
    level, where an ``M1`` inside a ``.subckt`` body is another device that
    only shares the name. ``None`` keeps every scope. A ref the cards give
    twice maps to its last card.
    """
    return {
        c.name.lower(): c
        for c in cards
        if c.kind == "instance" and c.name and (scope is None or c.scope == scope)
    }


# ---------------------------------------------------------------------------
# .SUBCKT
# ---------------------------------------------------------------------------


@dataclass
class SubcktCard:
    """Typed view over a ``.SUBCKT NAME P1 P2 [PARAMS: K=V ...]`` opener.

    Body cards live in the flat ``lex()`` list with ``scope=(name, ...)``;
    use ``ltspice_mcp.lib.spice_lex.iter_body`` to enumerate them.

    Setters on this view operate on the opener card alone. Cross-card
    operations like renaming the subcircuit (which must also update the
    matching ``.ENDS`` and the ``scope`` of every body card) live in
    ``spice_lex_ops`` since they need the full cards list.
    """

    card: SpiceCard
    name: str
    ports: list[str]
    _defaults: _Params = field(default_factory=_Params)
    _name_token: Token | None = None
    _used_params_marker: bool = False

    @classmethod
    def from_card(cls, card: SpiceCard) -> SubcktCard:
        _require_kind(card, "subckt")
        tokens = tokenize_body(card.body)
        if len(tokens) < 2:
            raise _malformed(card, "malformed .SUBCKT")
        name_token = tokens[1]
        name = name_token.text
        ports: list[str] = []
        defaults: list[Token] = []
        in_params = False
        used_marker = False
        for tok in tokens[2:]:
            if tok.kind == TokenKind.COMMENT_TRAIL:
                break
            if (
                tok.kind == TokenKind.BARE
                and tok.text.lower().rstrip(":") == "params"
                and tok.text.endswith(":")
            ):
                in_params = True
                used_marker = True
                continue
            if in_params:
                if tok.kind == TokenKind.KEY_VALUE:
                    defaults.append(tok)
            else:
                if tok.kind == TokenKind.BARE:
                    ports.append(tok.text)
                elif tok.kind == TokenKind.KEY_VALUE:
                    # ``.SUBCKT NAME P1 K=V`` — implicit PARAMS:; many
                    # SPICE flavours accept this without the marker.
                    in_params = True
                    defaults.append(tok)
        return cls(
            card=card,
            name=name,
            ports=ports,
            # A default is edited by rendering the opener again, as before.
            _defaults=_Params.of(defaults, in_place=False),
            _name_token=name_token,
            _used_params_marker=used_marker,
        )

    def set_name_local(self, new_name: str) -> None:
        """Rename the subcircuit on the **opener line only**.

        Does NOT update the matching ``.ENDS`` or the ``scope`` of body
        cards — for that, use ``spice_lex_ops.rename_subckt(cards, ...)``.
        Format-preserving when possible.
        """
        self.name = new_name
        tok = self._name_token
        if tok is not None and tok.body_offset >= 0:
            try:
                self.card.replace_span(tok.body_offset, tok.body_end, new_name)
                self._name_token = Token(
                    kind=tok.kind,
                    text=new_name,
                    body_offset=tok.body_offset,
                    body_length=len(new_name),
                )
                return
            except SpiceLexError:
                pass
        self._canonical_rerender()

    def set_ports(self, new_ports: list[str]) -> None:
        """Replace the port list. Canonical rerender."""
        self.ports = list(new_ports)
        self._canonical_rerender()

    @property
    def param_defaults(self) -> dict[str, str]:
        """The defaults by name as written, one value each; ``default_pairs``
        has every copy."""
        return self._defaults.as_dict()

    def default_pairs(self) -> list[tuple[str, str]]:
        """Every default on the opener, in order, a name given twice listed twice."""
        return list(self._defaults.pairs)

    def repeated_params(self) -> dict[str, list[str]]:
        """Each default the opener gives more than once (``_Params.repeated``)."""
        return self._defaults.repeated()

    def set_param_default(self, key: str, value: float | str) -> None:
        """Set every copy of a parameter default, or add it. Canonical
        rerender, every copy kept."""
        self._defaults.set(self.card, key, _format_value(value))
        self._canonical_rerender()

    def remove_param_default(self, key: str) -> None:
        """Remove every copy of a parameter default (no-op if absent)."""
        if self._defaults.remove(key):
            self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        parts: list[str] = [".SUBCKT", self.name, *self.ports]
        if self._defaults.pairs:
            if self._used_params_marker:
                parts.append("PARAMS:")
            parts.extend(self._defaults.rendered())
        body = " ".join(parts)
        self.card.replace_body(body)
        self._name_token = None


# ---------------------------------------------------------------------------
# Probes and .SAVE
# ---------------------------------------------------------------------------

#: A ``V(...)`` or ``I(...)`` probe in a card's text, its argument up to the
#: first closing parenthesis. The one probe pattern for a deck's cards: every
#: reader of the probes a card names goes through ``probe_refs``.
PROBE_REF_RE = re.compile(r"\b[VI]\s*\(([^()]*)\)", re.IGNORECASE)


@dataclass(frozen=True)
class ProbeRef:
    """One ``V(...)`` or ``I(...)`` probe a card names.

    ``text`` is the probe as written (``V(a, b)``) and ``kind`` its letter,
    ``V`` or ``I``. ``names`` are the node or device names inside it, a node
    pair split at its comma (``("a", "b")``), each stripped, an empty one left
    out. A reader that compares probes as signals compares ``text``; one that
    asks which nodes a card reaches reads ``names``.
    """

    text: str
    kind: str
    names: tuple[str, ...]


def probe_refs(card: SpiceCard) -> list[ProbeRef]:
    """Every ``V(...)``/``I(...)`` probe in a card, in order: in a ``.meas``,
    a ``.save`` or another directive, and in an instance's expression
    (``B1 x 0 V=V(a)*2``). Read from the lexed card, so a probe on a ``+``
    line is found and one in a comment is not."""
    return [
        ProbeRef(
            text=match.group(0),
            kind=match.group(0)[0].upper(),
            names=tuple(name for part in match.group(1).split(",") if (name := part.strip())),
        )
        for match in PROBE_REF_RE.finditer(card.body)
    ]


#: ``.save all`` or ``.save *``: the card saves every signal.
_SAVES_ALL_RE = re.compile(r"(?i)(?:^|\s)\.save\s+(?:all\b|\*)")


def probe_is_pattern(probe: ProbeRef) -> bool:
    """Whether a probe names its signals by a pattern (``V(*)``, ``I(R*)``,
    ``V(x?)``) rather than one by one. Which signals such a ``.save`` saves is
    not recorded for either LTspice build (``inputs/cases.toml``, behaviour
    ``save-pattern``), so a reader of a ``.save`` that cannot tell says so
    rather than read the pattern as a name."""
    return any(mark in name for name in probe.names for mark in "*?")


@dataclass
class SaveCard:
    """Typed view over a ``.SAVE`` directive: the probes it saves, or that it
    saves every signal (``.save all``). A probe may name its signals by a
    pattern (``probe_is_pattern``). Read-only."""

    card: SpiceCard
    probes: list[ProbeRef]
    saves_all: bool

    @classmethod
    def from_card(cls, card: SpiceCard) -> SaveCard:
        _require_kind(card, "directive")
        if not _is_save(card):
            raise _malformed(card, "not a .save card")
        return cls(
            card=card,
            probes=probe_refs(card),
            saves_all=_SAVES_ALL_RE.search(card.body) is not None,
        )


def _is_save(card: SpiceCard) -> bool:
    return directive_head(card) == ".save"


def save_cards(cards: Sequence[SpiceCard]) -> list[SaveCard]:
    """The ``.save`` directives in ``cards``, each as a ``SaveCard``."""
    return [
        SaveCard.from_card(card) for card in cards if card.kind == "directive" and _is_save(card)
    ]


# ---------------------------------------------------------------------------
# .MEAS
# ---------------------------------------------------------------------------


# Mirrors ``spice_lex.MEAS_ANALYSIS_TOKENS`` at the type level (a Literal cannot
# be derived from the runtime constant — keep the two in sync).
MeasAnalysis = Literal["tran", "ac", "dc", "op", "noise", "sp"]


@dataclass
class FunctionCall:
    """One ``name(...)`` invocation inside a ``.MEAS`` body."""

    name: str
    args_text: str  # raw text inside the parens, no surrounding ()


@dataclass
class MeasCard:
    """Typed view over a ``.MEAS [analysis] NAME ...`` directive.

    Exposes its body as classified tokens (``body_tokens``) plus
    pre-extracted ``function_calls`` and ``signal_refs`` (the card's probes,
    as ``probe_refs`` reads them) so callers (esp. ``spice_validator``) can
    run semantic checks without re-parsing characters. Read-only.
    """

    card: SpiceCard
    analysis: MeasAnalysis | None
    name: str
    body_tokens: list[Token]
    function_calls: list[FunctionCall]
    signal_refs: list[ProbeRef]

    @classmethod
    def from_card(cls, card: SpiceCard) -> MeasCard:
        _require_kind(card, "meas")
        tokens = tokenize_body(card.body)
        # tokens[0] = ".MEAS" (or ".MEASURE"). tokens[1] is either
        # analysis kind or measurement label.
        if len(tokens) < 2:
            raise _malformed(card, "malformed .MEAS")
        idx = 1
        analysis: MeasAnalysis | None = None
        second = tokens[idx].text.lower()
        if second in MEAS_ANALYSIS_TOKENS:
            analysis = second  # type: ignore[assignment]
            idx += 1
        if idx >= len(tokens):
            raise _malformed(card, ".MEAS missing label")
        meas_name = tokens[idx].text
        body_tokens = list(tokens[idx + 1 :])

        # Function-call detection: BARE immediately followed by PARENED.
        # Recurses into PARENED args so nested calls are also surfaced
        # (``mag(V(out))`` reports both ``mag`` and ``V``).
        function_calls: list[FunctionCall] = []
        _collect_function_calls(body_tokens, function_calls)

        return cls(
            card=card,
            analysis=analysis,
            name=meas_name,
            body_tokens=body_tokens,
            function_calls=function_calls,
            signal_refs=probe_refs(card),
        )


# ---------------------------------------------------------------------------
# Convenience iterators
# ---------------------------------------------------------------------------


def iter_models(
    cards: Sequence[SpiceCard],
    scope: tuple[str, ...] | None = None,
) -> Iterator[ModelCard]:
    """Iterate ``ModelCard`` views over every ``.MODEL`` card.

    ``scope=None`` covers all scopes; ``scope=()`` is top level only.
    """
    for card in cards:
        if card.kind != "model":
            continue
        if scope is not None and card.scope != scope:
            continue
        yield ModelCard.from_card(card)


def iter_instances(
    cards: Sequence[SpiceCard],
    prefix: str | None = None,
) -> Iterator[InstanceLine]:
    """Iterate ``InstanceLine`` views over every instance card.

    ``prefix`` filters by the first letter of the ref (case-insensitive),
    e.g. ``prefix="M"`` for MOSFETs only.
    """
    p = prefix.upper() if prefix else None
    for card in cards:
        if card.kind != "instance":
            continue
        if p is not None and (not card.name or card.name[:1].upper() != p):
            continue
        yield InstanceLine.from_card(card)


def find_model(
    cards: Sequence[SpiceCard],
    name: str,
    scope: tuple[str, ...] = (),
) -> ModelCard | None:
    """Find a ``.MODEL`` card by name, walking outward from ``scope``.

    SPICE name resolution: search the current scope, then each enclosing
    scope, ending at top level. Returns the first hit (case-insensitive
    name match), or None. Single pass over ``cards``: collects all model
    cards matching ``name`` keyed by scope, then probes outward.
    """
    target = name.lower()
    by_scope: dict[tuple[str, ...], SpiceCard] = {}
    for card in cards:
        if card.kind != "model" or not card.name:
            continue
        if card.name.lower() != target:
            continue
        # First match per scope wins (matches LTspice's first-definition
        # behaviour within a scope).
        by_scope.setdefault(card.scope, card)
    for depth in range(len(scope), -1, -1):
        hit = by_scope.get(scope[:depth])
        if hit is not None:
            return ModelCard.from_card(hit)
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shift_tokens(tokens: list[Token], edit_old_end: int, delta: int) -> list[Token]:
    """``tokens`` with every one at or past ``edit_old_end`` moved by ``delta``."""
    return [
        replace(tok, body_offset=tok.body_offset + delta)
        if delta and tok.body_offset >= edit_old_end
        else tok
        for tok in tokens
    ]


def _shift_single_token(tok: Token | None, edit_old_end: int, delta: int) -> Token | None:
    """``_shift_tokens`` for a single optional token."""
    if tok is None or delta == 0:
        return tok
    if tok.body_offset < edit_old_end:
        return tok
    return Token(
        kind=tok.kind,
        text=tok.text,
        key=tok.key,
        value=tok.value,
        body_offset=tok.body_offset + delta,
        body_length=tok.body_length,
    )


#: The function name at the end of a BARE token that a PARENED token follows.
_CALL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _collect_function_calls(
    tokens: Sequence[Token],
    out_calls: list[FunctionCall],
) -> None:
    """Walk ``tokens`` and append any ``BARE PARENED`` function calls.

    Recurses into the contents of each PARENED token so nested calls
    are surfaced too (``mag(V(out))`` yields both ``mag`` and ``V``).
    Also recurses into BRACED contents — ``{vdb(out)}`` should still
    surface ``vdb`` for validator rules.

    The tokenizer keeps an expression's operators in its BARE tokens, so the
    call in ``V(s)*cos(x)`` sits behind ``*cos``; the name is the identifier
    the token ends with. A ``key=value`` token holds an expression in its
    value (``time=cos(1)``, ``TD={atan(1)}``), which is walked the same way.
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < n else None
        if (
            t.kind == TokenKind.BARE
            and nxt is not None
            and nxt.kind == TokenKind.PARENED
            and (called := _CALL_NAME_RE.search(t.text))
        ):
            out_calls.append(FunctionCall(name=called.group(0), args_text=nxt.text[1:-1]))
        # Recurse into PARENED / BRACED contents and into a key=value's value.
        if t.kind in (TokenKind.PARENED, TokenKind.BRACED):
            inner_text = t.text[1:-1]
        elif t.kind == TokenKind.KEY_VALUE and t.value:
            inner_text = t.value
        else:
            continue
        try:
            inner_tokens = tokenize_body(inner_text)
        except SpiceLexError:
            continue
        _collect_function_calls(inner_tokens, out_calls)
