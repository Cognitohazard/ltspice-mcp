"""Element-class-typed dispatcher for ``set_component_value``.

Routes user input to the right slot per element class (R/C/L positional
value, V/I multi-token source spec, B-source ``V=``/``I=`` prefix,
E/G/F/H positional gain, M/Q/J/X model+params) using ``spice_lex_views``
typed views for surgical edits. The previous spicelib-editor path
treated every element as a free-form line, which corrupted B-sources
(stripped ``V=``), E-sources (replaced 3 fields with 1), and rejected
PULSE/SIN waveforms.

Public surface: ``apply_value_to_instance(card, raw_value) -> ApplyResult``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.format import is_scaled_number
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import InstanceLine, body_has_stray_kv_remnant

#: The token kinds that stand in a positional slot of an element card; with
#: ``KEY_VALUE`` they are every kind a well-formed value is made of.
POSITIONAL_KINDS = (
    TokenKind.BARE,
    TokenKind.QUOTED,
    TokenKind.BRACED,
    TokenKind.PARENED,
)


@dataclass
class ApplyResult:
    reference: str
    old_summary: str
    new_summary: str


@dataclass
class _ApplyCtx:
    card: SpiceCard
    ref: str
    raw_value: str
    body_tokens: list[Token]
    user_pos: list[Token]
    user_kv: list[Token]


def apply_value_to_instance(card: SpiceCard, raw_value: str) -> ApplyResult:
    """Apply user-supplied value to an instance card via typed dispatch.

    Mutates ``card`` in place. Returns a before/after summary for
    diagnostic messages. Raises ``NetlistError`` on shapes that don't
    match the element class.
    """
    if card.kind != "instance":
        raise NetlistError(f"expected instance card, got {card.kind}")

    body_tokens = [t for t in tokenize_body(card.body) if t.kind != TokenKind.COMMENT_TRAIL]
    if not body_tokens or body_tokens[0].kind != TokenKind.BARE:
        raise NetlistError("malformed instance card body")
    ref = body_tokens[0].text
    prefix = ref[:1].upper()

    if not raw_value or not raw_value.strip():
        raise NetlistError(f"Component {ref!r} value must not be empty")
    if "\n" in raw_value or "\r" in raw_value:
        raise NetlistError(f"Component {ref!r} value must be a single line; got embedded newline")

    try:
        user_tokens = [t for t in tokenize_body(raw_value) if t.kind != TokenKind.COMMENT_TRAIL]
    except SpiceLexError as e:
        raise NetlistError(f"Component {ref!r} value {raw_value!r} failed to parse: {e}") from e
    if not user_tokens:
        raise NetlistError(f"Component {ref!r} value parsed to no tokens")
    if any(t.kind not in (*POSITIONAL_KINDS, TokenKind.KEY_VALUE) for t in user_tokens):
        raise NetlistError(
            f"Component {ref!r} value {raw_value!r} contains an "
            "unrecognised token (stray equals sign or unbalanced quote)."
        )
    # A source's value may go on past a KEY=VALUE word (a PWL's
    # ``TIME_SCALE_FACTOR=2 (0 4 1 4)``), which read_source_text reads.
    if prefix not in SOURCE_PARAMETERS:
        seen_kv = False
        for tok in user_tokens:
            if tok.kind == TokenKind.KEY_VALUE:
                seen_kv = True
            elif seen_kv and tok.kind in POSITIONAL_KINDS:
                raise NetlistError(
                    f"Component {ref!r} value {raw_value!r} contains trailing tokens "
                    "after a KEY=VALUE assignment. Wrap expressions in braces or "
                    "write function-call values without separating the function name "
                    "from its parentheses."
                )

    handler = _DISPATCH.get(prefix)
    if handler is None:
        if prefix == "K":
            raise NetlistError(
                f"set_component_value does not support mutual-inductance "
                f"K elements (got {ref!r}). Edit the card directly."
            )
        raise NetlistError(
            f"set_component_value does not support {prefix!r}-prefixed element "
            f"{ref!r}. Edit the card directly."
        )

    ctx = _ApplyCtx(
        card=card,
        ref=ref,
        raw_value=raw_value,
        body_tokens=body_tokens,
        user_pos=[t for t in user_tokens if t.kind in POSITIONAL_KINDS],
        user_kv=[t for t in user_tokens if t.kind == TokenKind.KEY_VALUE],
    )
    return handler(ctx)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _positional_after_ref(body_tokens: list[Token]) -> list[Token]:
    """Positional tokens after the leading ref, stopping at the first KV."""
    out: list[Token] = []
    for tok in body_tokens[1:]:
        if tok.kind == TokenKind.KEY_VALUE:
            break
        if tok.kind in POSITIONAL_KINDS:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# R / C / L — single positional value + optional KV params
# ---------------------------------------------------------------------------


def _apply_passive(ctx: _ApplyCtx) -> ApplyResult:
    if len(ctx.user_pos) > 1:
        raise NetlistError(
            f"Component {ctx.ref!r} value {ctx.raw_value!r} contains whitespace "
            f"that the parser couldn't fold into a single value token "
            f"(got {len(ctx.user_pos)} positional tokens). Wrap SPICE "
            "expressions in braces ({...}) or use the parameter form "
            "(e.g. 'NMOS1 W=10u L=1u'); a bare space-separated value "
            "would corrupt the netlist line."
        )
    inst = InstanceLine.from_card(ctx.card)
    old = inst.value or ""
    new_value = ctx.user_pos[0].text if ctx.user_pos else None
    if new_value is not None:
        inst.set_value(new_value)
    for kv in ctx.user_kv:
        assert kv.key is not None and kv.value is not None
        inst.set_param(kv.key, kv.value)
    return ApplyResult(ctx.ref, old, new_value or (inst.value or ""))


# ---------------------------------------------------------------------------
# V / I — exactly 2 nodes; then a multi-token value and the source's own
# parameters
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
    parts, cursor = [], 0
    for start, stop in spans:
        parts.append(text[cursor:start].strip())
        cursor = stop
    parts.append(text[cursor : len(text) if end is None else end].strip())
    return " ".join(part for part in parts if part)


def source_word_key(word: str) -> str:
    """The name a source's own parameter goes by: its key (``AC=2`` and
    ``AC 2`` alike), or the flag itself."""
    return word.split("=", 1)[0].split(None, 1)[0].strip().casefold()


def _is_amount(token: Token) -> bool:
    """A word that can be an AC magnitude or phase: a number or a braced expression."""
    return token.kind == TokenKind.BRACED or (
        token.kind == TokenKind.BARE and is_scaled_number(token.text)
    )


def read_source_text(text: str, prefix: str) -> SourceText:
    """``text``, the words after a source's nodes (on a card, or in a sheet
    part's Value), read as its value and its own parameters
    (``SOURCE_PARAMETERS``, ``AC`` with its magnitude and any phase, and a
    current source's ``load``): the one rule every edit of a source's value
    keeps them by.

    ``InstanceLine``, which display, comparison and the nested-instance edits
    read a card through, takes a current source's ``load`` as part of its
    value, and does not read a PWL with a scale keyword before its points,
    which both builds run (``deck/source_parameters``); a test whose name says
    so pins both differences.

    Raises ``SpiceLexError`` for text that does not split into words.
    """
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
    played = None
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        key = (tok.key or "").casefold() if tok.kind == TokenKind.KEY_VALUE else ""
        bare = tok.text.casefold() if tok.kind == TokenKind.BARE else ""
        if key in own_keys or (letter == "I" and bare == LOAD_FLAG):
            own.append((tok.body_offset, tok.body_end))
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
    return SourceText(
        value=text_without(text, own, end),
        own=tuple(" ".join(text[start:stop].split()) for start, stop in own),
        played=played,
    )


def edited_source(
    reference: str, old: SourceText, new: SourceText, *, add_keyed: bool = True
) -> SourceText:
    """The source text an edit gives a source that held ``old``: ``new``'s
    value, or ``old``'s where ``new`` is keyed parameters alone (``Rser=2``),
    which is an edit of those; ``old``'s own parameters with every copy of one
    ``new`` names replaced; then the ones ``new`` adds. A ``new`` with any
    other word, an ``AC 1`` or a ``load`` among them, is the whole value.

    ``add_keyed`` False leaves out an added ``KEY=VALUE`` parameter: a sheet
    part sets that where it already gives the key, in any of its attributes
    (``schematic_ops._set_source_value``), where a card has only one place.

    A source that plays a .wav file has no value, so a value for one is
    refused, naming what to change.
    """
    if old.played is not None and new.value and new.played is None:
        raise NetlistError(
            f"Component {reference!r} plays a .wav file ({old.played}), which sets "
            "its level, so it has no value to replace. Remove the wavefile= (and any "
            "chan=) from it to give it a value, or name another file there."
        )
    given = {source_word_key(word): word for word in new.own}
    held = {source_word_key(word) for word in old.own}
    own = [given.get(source_word_key(word), word) for word in old.own]
    own += [
        word for key, word in given.items() if key not in held and (add_keyed or "=" not in word)
    ]
    keyed_alone = not new.value and all("=" in word for word in new.own)
    value = old.value if keyed_alone else new.value
    return SourceText(value, tuple(own), new.played or old.played)


def _apply_indep_source(ctx: _ApplyCtx) -> ApplyResult:
    pos = _positional_after_ref(ctx.body_tokens)
    if len(pos) < 2:
        raise NetlistError(
            f"Component {ctx.ref!r}: existing card has fewer than 2 nodes; "
            "can't determine where the value field starts."
        )
    n1, n2 = pos[0].text, pos[1].text
    letter = ctx.ref[:1]
    old = read_source_text(ctx.card.body[pos[1].body_end :], letter)
    # The value was split into words above, so this reading of it cannot fail.
    edited = edited_source(ctx.ref, old, read_source_text(ctx.raw_value, letter))
    ctx.card.replace_body(" ".join(w for w in (ctx.ref, n1, n2, edited.text) if w))
    return ApplyResult(ctx.ref, old.text, edited.text)


# ---------------------------------------------------------------------------
# B — behavioural; body is ``Bx n1 n2 V=expr`` or ``I=expr``
# ---------------------------------------------------------------------------


def _apply_b_source(ctx: _ApplyCtx) -> ApplyResult:
    # An unbraced expression with whitespace around its operators
    # (``V = V(a) + V(b)``) parses to a first key=value plus orphan tokens the
    # editor can't see. Rewriting the key=value span would leave those orphans
    # behind and corrupt the card, so refuse rather than silently mangle it.
    if body_has_stray_kv_remnant(ctx.card.body):
        raise NetlistError(
            f"Component {ctx.ref!r}: its behavioural expression is not fully "
            "parseable (orphan tokens after the first key=value). Wrap the "
            "expression in braces (e.g. V={...}) in the netlist, then edit it."
        )
    inst = InstanceLine.from_card(ctx.card)
    existing_upper = {k.upper(): v for k, v in inst.params.items()}
    has_v, has_i = "V" in existing_upper, "I" in existing_upper

    if ctx.user_kv:
        old_repr = " ".join(f"{k}={v}" for k, v in inst.params.items())
        new_upper = {kv.key.upper() for kv in ctx.user_kv if kv.key is not None}
        for kv in ctx.user_kv:
            assert kv.key is not None and kv.value is not None
            inst.set_param(kv.key, kv.value)
        # Switching V↔I drops the opposite — B-sources carry one type at a time.
        if "V" in new_upper and has_i and "I" not in new_upper:
            inst.remove_param("I")
        if "I" in new_upper and has_v and "V" not in new_upper:
            inst.remove_param("V")
        new_summary = " ".join(f"{kv.key}={kv.value}" for kv in ctx.user_kv)
        return ApplyResult(ctx.ref, old_repr, new_summary)

    if len(ctx.user_pos) != 1:
        raise NetlistError(
            f"Component {ctx.ref!r}: behavioural sources need a single "
            "expression value (a brace expression or constant)."
        )
    if has_v and has_i:
        raise NetlistError(
            f"Component {ctx.ref!r} has both V= and I= already; pass "
            "the value as 'V=...' or 'I=...' explicitly."
        )
    prefix = "V" if has_v else "I" if has_i else None
    if prefix is None:
        raise NetlistError(
            f"Component {ctx.ref!r} has no V=/I= prefix in its existing card; "
            "pass the value as 'V=expr' or 'I=expr' explicitly to set the "
            "source type."
        )
    expr_text = ctx.user_pos[0].text
    inst.set_param(prefix, expr_text)
    return ApplyResult(ctx.ref, f"{prefix}={existing_upper[prefix]}", f"{prefix}={expr_text}")


# ---------------------------------------------------------------------------
# E / G / F / H — controlled sources
#   - Positional gain form: ``E1 buf 0 in 0 10`` / ``F1 out 0 V_sense 2``.
#     The trailing positional is the gain and is what InstanceLine.value
#     points at, so set_value() updates only the gain.
#   - KV form (POLY/VALUE=/TABLE): user passes KEY=VALUE.
#   - F/H also accept ``"ctrl_ref gain"`` to swap the control source ref.
# ---------------------------------------------------------------------------


def _apply_controlled_source(ctx: _ApplyCtx, *, allow_two_positional: bool) -> ApplyResult:
    inst = InstanceLine.from_card(ctx.card)

    if ctx.user_kv and not ctx.user_pos:
        old_repr = " ".join(f"{k}={v}" for k, v in inst.params.items())
        for kv in ctx.user_kv:
            assert kv.key is not None and kv.value is not None
            inst.set_param(kv.key, kv.value)
        return ApplyResult(
            ctx.ref,
            old_repr,
            " ".join(f"{kv.key}={kv.value}" for kv in ctx.user_kv),
        )

    if len(ctx.user_pos) == 1 and not ctx.user_kv:
        if inst.value is None:
            raise NetlistError(
                f"Component {ctx.ref!r} appears to be in KV form (POLY/VALUE=); "
                "pass the new value as a KEY=VALUE token instead of a bare number."
            )
        new_gain = ctx.user_pos[0].text
        old_gain = inst.value
        inst.set_value(new_gain)
        return ApplyResult(ctx.ref, old_gain, new_gain)

    if allow_two_positional and len(ctx.user_pos) == 2 and not ctx.user_kv:
        pos = _positional_after_ref(ctx.body_tokens)
        if len(pos) < 3:
            raise NetlistError(
                f"Component {ctx.ref!r}: existing card lacks the control-source ref slot."
            )
        n1, n2 = pos[0].text, pos[1].text
        ctrl, gain = ctx.user_pos[0].text, ctx.user_pos[1].text
        old_summary = " ".join(t.text for t in pos[2:])
        ctx.card.replace_body(f"{ctx.ref} {n1} {n2} {ctrl} {gain}")
        return ApplyResult(ctx.ref, old_summary, f"{ctrl} {gain}")

    if allow_two_positional:
        raise NetlistError(
            f"Component {ctx.ref!r}: F/H source values are either a single "
            f"gain number, 'ctrl_ref gain', or KEY=VALUE params. Got {ctx.raw_value!r}."
        )
    raise NetlistError(
        f"Component {ctx.ref!r}: pass either a single gain value (e.g. '20') OR "
        f"KEY=VALUE params (e.g. 'POLY(1) (in,0) 0 1') — not both. "
        f"Got {ctx.raw_value!r}."
    )


def _apply_eg_source(ctx: _ApplyCtx) -> ApplyResult:
    return _apply_controlled_source(ctx, allow_two_positional=False)


def _apply_fh_source(ctx: _ApplyCtx) -> ApplyResult:
    return _apply_controlled_source(ctx, allow_two_positional=True)


# ---------------------------------------------------------------------------
# M / Q / J / X — model/subckt name head + KV params
# ---------------------------------------------------------------------------


def _apply_head_with_kv(ctx: _ApplyCtx, *, slot_label: str) -> ApplyResult:
    if len(ctx.user_pos) > 1:
        raise NetlistError(
            f"Component {ctx.ref!r}: {slot_label} value takes a single "
            f"name (got {len(ctx.user_pos)} positional tokens in {ctx.raw_value!r})."
        )
    inst = InstanceLine.from_card(ctx.card)

    # Build a slot-aware diagnostic before mutating: a KV-only update
    # reports the changed param (``W=10u -> W=20u``) rather than the
    # untouched model name.
    old_parts: list[str] = []
    new_parts: list[str] = []
    new_head: str | None = None
    if ctx.user_pos:
        new_head = ctx.user_pos[0].text.strip('"')
        old_parts.append(inst.model or "")
        new_parts.append(new_head)
    for kv in ctx.user_kv:
        assert kv.key is not None and kv.value is not None
        old_kv = next((v for k, v in inst.params.items() if k.lower() == kv.key.lower()), "")
        old_parts.append(f"{kv.key}={old_kv}")
        new_parts.append(f"{kv.key}={kv.value}")

    if new_head is not None:
        inst.set_model(new_head)
    for kv in ctx.user_kv:
        assert kv.key is not None and kv.value is not None
        inst.set_param(kv.key, kv.value)

    return ApplyResult(
        ctx.ref,
        " ".join(old_parts) or (inst.model or ""),
        " ".join(new_parts) or (new_head or ""),
    )


def _apply_active_device(ctx: _ApplyCtx) -> ApplyResult:
    return _apply_head_with_kv(ctx, slot_label="device")


def _apply_subckt(ctx: _ApplyCtx) -> ApplyResult:
    return _apply_head_with_kv(ctx, slot_label="X-instance")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Callable[[_ApplyCtx], ApplyResult]] = {
    "R": _apply_passive,
    "C": _apply_passive,
    "L": _apply_passive,
    "V": _apply_indep_source,
    "I": _apply_indep_source,
    "B": _apply_b_source,
    "E": _apply_eg_source,
    "G": _apply_eg_source,
    "F": _apply_fh_source,
    "H": _apply_fh_source,
    "M": _apply_active_device,
    "Q": _apply_active_device,
    "J": _apply_active_device,
    # Diode and the voltage/current-controlled switches carry a trailing model
    # name — the same card shape as M/Q/J, so set_component_value swaps the
    # model/part number the same way.
    "D": _apply_active_device,
    "S": _apply_active_device,
    "W": _apply_active_device,
    "X": _apply_subckt,
}
