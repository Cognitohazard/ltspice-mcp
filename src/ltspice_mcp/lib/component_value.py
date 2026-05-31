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
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import InstanceLine

_POSITIONAL_KINDS = (
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
    if any(t.kind not in (*_POSITIONAL_KINDS, TokenKind.KEY_VALUE) for t in user_tokens):
        raise NetlistError(
            f"Component {ref!r} value {raw_value!r} contains an "
            "unrecognised token (stray equals sign or unbalanced quote)."
        )
    seen_kv = False
    for tok in user_tokens:
        if tok.kind == TokenKind.KEY_VALUE:
            seen_kv = True
        elif seen_kv and tok.kind in _POSITIONAL_KINDS:
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
        raise NetlistError(f"Unsupported element prefix {prefix!r} for {ref!r}")

    ctx = _ApplyCtx(
        card=card,
        ref=ref,
        raw_value=raw_value,
        body_tokens=body_tokens,
        user_pos=[t for t in user_tokens if t.kind in _POSITIONAL_KINDS],
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
        if tok.kind in _POSITIONAL_KINDS:
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
# V / I — exactly 2 nodes; everything after is a multi-token value spec
# ---------------------------------------------------------------------------


def _apply_indep_source(ctx: _ApplyCtx) -> ApplyResult:
    if ctx.user_kv:
        raise NetlistError(
            f"Component {ctx.ref!r}: independent sources don't take KEY=VALUE "
            f"params via set_component_value (got {ctx.raw_value!r}). The DC/AC "
            "magnitude and source spec are positional."
        )
    if not ctx.user_pos:
        raise NetlistError(f"Component {ctx.ref!r}: empty value")
    pos = _positional_after_ref(ctx.body_tokens)
    if len(pos) < 2:
        raise NetlistError(
            f"Component {ctx.ref!r}: existing card has fewer than 2 nodes; "
            "can't determine where the value field starts."
        )
    n1, n2 = pos[0].text, pos[1].text
    new_value = ctx.raw_value.strip()
    old_value = " ".join(t.text for t in pos[2:])
    ctx.card.replace_body(f"{ctx.ref} {n1} {n2} {new_value}")
    return ApplyResult(ctx.ref, old_value, new_value)


# ---------------------------------------------------------------------------
# B — behavioural; body is ``Bx n1 n2 V=expr`` or ``I=expr``
# ---------------------------------------------------------------------------


def _apply_b_source(ctx: _ApplyCtx) -> ApplyResult:
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
    "X": _apply_subckt,
}
