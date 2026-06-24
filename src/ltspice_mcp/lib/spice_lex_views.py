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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ltspice_mcp.lib.format import format_spice_value as _format_value
from ltspice_mcp.lib.spice_lex import (
    MEAS_ANALYSIS_TOKENS,
    SpiceCard,
    SpiceLexError,
    SpiceLexErrorCategory,
    Token,
    TokenKind,
    _strip_matching_quotes,
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


# ---------------------------------------------------------------------------
# .MODEL
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Typed view over a ``.MODEL NAME TYPE(...)`` card.

    ``params`` carries the raw value tokens (``"0.7"``, ``"{vto_n}"``,
    ``"100u"``); callers parse to numeric as needed. Keys are stored
    case-preserved; matching against them folds to lower at lookup.
    """

    card: SpiceCard
    name: str
    type: str
    level: int | None
    params: dict[str, str] = field(default_factory=dict)
    # Map of (case-preserved key) → KEY_VALUE Token for in-place edits.
    _param_tokens: dict[str, Token] = field(default_factory=dict)

    @classmethod
    def from_card(cls, card: SpiceCard) -> ModelCard:
        _require_kind(card, "model")
        tokens = tokenize_body(card.body)
        # Expected shape: BARE(.MODEL) BARE(name) (BARE|QUOTED)(type)
        # PARENED(params) | KEY_VALUE...
        if len(tokens) < 3:
            raise _malformed(card, "malformed .MODEL card")
        # tokens[0] is ".MODEL" (BARE), tokens[1] is name, tokens[2] is type.
        name = _strip_matching_quotes(tokens[1].text)
        type_ = _strip_matching_quotes(tokens[2].text)
        params: dict[str, str] = {}
        param_tokens: dict[str, Token] = {}
        # Params can come as a single PARENED group or as a sequence of
        # KEY_VALUE tokens after the type. PARENED params don't have
        # individual body offsets (they're nested), so they get
        # canonical-form rerender on edit.
        for tok in tokens[3:]:
            if tok.kind == TokenKind.PARENED:
                inner = tok.text[1:-1]  # strip parens
                for inner_tok in tokenize_body(inner):
                    if inner_tok.kind == TokenKind.KEY_VALUE:
                        assert inner_tok.key is not None
                        assert inner_tok.value is not None
                        params[inner_tok.key] = inner_tok.value
                        # No top-level body_offset for nested params —
                        # leave _param_tokens empty for these.
            elif tok.kind == TokenKind.KEY_VALUE:
                assert tok.key is not None
                assert tok.value is not None
                params[tok.key] = tok.value
                param_tokens[tok.key] = tok
            elif tok.kind == TokenKind.COMMENT_TRAIL:
                break
        level = None
        for k, v in params.items():
            if k.lower() == "level":
                try:
                    level = int(v)
                except ValueError:
                    level = None
                break
        return cls(
            card=card,
            name=name,
            type=type_,
            level=level,
            params=params,
            _param_tokens=param_tokens,
        )

    def set_param(self, key: str, value: float | str) -> None:
        """Set or add a parameter, flipping the card dirty."""
        new = _format_value(value)
        for existing in list(self.params):
            if existing.lower() == key.lower():
                self.params[existing] = new
                self._update_param_in_place(existing, new)
                return
        # New param — fall back to canonical rerender (we don't know
        # where to insert in the original layout).
        self.params[key] = new
        self._canonical_rerender()

    def remove_param(self, key: str) -> None:
        """Remove a parameter (no-op if absent)."""
        for existing in list(self.params):
            if existing.lower() == key.lower():
                del self.params[existing]
                self._param_tokens.pop(existing, None)
                self._canonical_rerender()
                return

    def set_name(self, new_name: str) -> None:
        """Rename the model. Canonical rerender."""
        self.name = new_name
        self._canonical_rerender()
        self.card.name = new_name

    def _update_param_in_place(self, key: str, new_value: str) -> None:
        """Try format-preserving replacement; fall back to canonical.

        On success, refreshes the edited token AND shifts every other
        cached param token whose body_offset sat past the edit by the
        length delta. Without the shift, a second edit on a different
        key splices into stale offsets and corrupts the card body —
        e.g. ``.MODEL D1 D IS=1e-14 N=1`` after sequential IS / N
        edits would emit ``IS=2.3456N=1.54 N=1``.
        """
        tok = self._param_tokens.get(key)
        if tok is not None and tok.body_offset >= 0 and tok.value is not None:
            new_token_text = f"{tok.key}={new_value}"
            old_body_end = tok.body_end
            delta = len(new_token_text) - tok.body_length
            try:
                self.card.replace_span(tok.body_offset, old_body_end, new_token_text)
                self._param_tokens[key] = Token(
                    kind=TokenKind.KEY_VALUE,
                    text=new_token_text,
                    key=tok.key,
                    value=new_value,
                    body_offset=tok.body_offset,
                    body_length=len(new_token_text),
                )
                _shift_cached_param_tokens(
                    self._param_tokens,
                    edit_old_end=old_body_end,
                    delta=delta,
                    exclude_key=key,
                )
                return
            except ValueError:
                pass  # fall through to canonical
        self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        params_text = " ".join(f"{k}={v}" for k, v in self.params.items())
        body = f".MODEL {self.name} {self.type}({params_text})"
        self.card.replace_body(body)
        # _param_tokens is now stale (different body offsets) — clear it
        # so subsequent edits go through canonical rerender too. Caller
        # who wants in-place edits after a structural change must
        # re-derive the view via from_card.
        self._param_tokens.clear()


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


@dataclass(frozen=True)
class ElementSpec:
    default_kind: ElementKind
    min_nodes: int
    kv_means_params_only: bool = False

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

# A switch (S/W) carries an optional ON/OFF state token AFTER its model name.
# It is never a model name, so it is peeled into the value tail before model
# identification — otherwise the last-positional heuristic would mistake the
# state for the model and clobber the real model on a set_model edit.
_SWITCH_STATES = frozenset({"on", "off"})


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

    - ``M`` / ``Q`` / ``J`` / ``X``: ``model`` carries the model or
      subckt name, ``value`` is None.
    - ``R`` / ``C`` / ``L`` / ``V`` / ``I``: ``value`` carries the
      passive value or source magnitude (e.g. ``"1k"``), ``model`` is
      None.
    - ``B`` / ``E`` / ``G``: both are None — the value lives in
      ``params["V"]`` or ``params["I"]``.
    - ``F`` / ``H``: ``model`` carries the controlling source name.
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
    params: dict[str, str] = field(default_factory=dict)
    _model_token: Token | None = None
    _param_tokens: dict[str, Token] = field(default_factory=dict)
    _value_param_key: str | None = None
    # Body span (start, end) covering exactly the positional node tokens, so
    # set_nodes can rewrite connectivity without disturbing the value/params
    # tail. None when there are no nodes or their offsets are synthesized.
    _node_span: tuple[int, int] | None = None

    @classmethod
    def from_card(cls, card: SpiceCard) -> InstanceLine:
        _require_kind(card, "instance")
        tokens = tokenize_body(card.body)
        if not tokens:
            raise _malformed(card, "empty instance card")
        if tokens[0].kind != TokenKind.BARE:
            raise _malformed(card, "instance card does not start with a bare ref")
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
            #  - a switch ON/OFF state (``S1 ... MYSW ON``).
            # Both follow the model and are preserved as the value tail.
            # (Variable-arity classes with a bare trailing NUMBER — e.g. a
            # MOSFET/BJT area factor — stay on the last-positional heuristic:
            # the trailing number is indistinguishable from a numerically-named
            # model like a ``555`` subckt without arity we don't have.)
            tail: list[Token] = []
            pos = positional
            # ON/OFF is a state ONLY for switches (S/W). For any other element a
            # trailing ON/OFF is the model/subckt name itself (e.g. ``D1 a k ON``),
            # so peeling it there would corrupt the parse.
            if prefix in ("S", "W") and pos[-1].text.strip('"').lower() in _SWITCH_STATES:
                tail = [pos[-1]]
                pos = pos[:-1]
            if pos:
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

        params: dict[str, str] = {}
        param_tokens: dict[str, Token] = {}
        if first_kv is not None:
            for tok in tokens[first_kv:]:
                if tok.kind == TokenKind.KEY_VALUE:
                    assert tok.key is not None
                    assert tok.value is not None
                    params[tok.key] = tok.value
                    param_tokens[tok.key] = tok
                elif tok.kind == TokenKind.COMMENT_TRAIL:
                    break

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
            params=params,
            _model_token=model_token,
            _param_tokens=param_tokens,
            _value_param_key=value_param_key,
            _node_span=node_span,
        )

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
        self._param_tokens.clear()

    def get_param(self, key: str) -> str | None:
        """Case-insensitive parameter read, matching the CI semantics of
        ``set_param``/``remove_param``. ``params`` preserves source casing
        (ngspice decks conventionally write lowercase ``w=``/``l=``), so a
        plain dict lookup silently misses — readers must come through here.
        """
        key_lower = key.lower()
        return next((v for k, v in self.params.items() if k.lower() == key_lower), None)

    def set_param(self, key: str, value: float | str) -> None:
        """Set or add a parameter, flipping the card dirty.

        On in-place success, also shifts every other cached param token
        AND the cached model token by the length delta so subsequent
        edits on the same view operate on correct offsets.
        """
        new = _format_value(value)
        for existing in list(self.params):
            if existing.lower() == key.lower():
                self.params[existing] = new
                tok = self._param_tokens.get(existing)
                if tok is not None and tok.body_offset >= 0:
                    new_text = f"{tok.key}={new}"
                    old_body_end = tok.body_end
                    delta = len(new_text) - tok.body_length
                    try:
                        self.card.replace_span(tok.body_offset, old_body_end, new_text)
                        self._param_tokens[existing] = Token(
                            kind=TokenKind.KEY_VALUE,
                            text=new_text,
                            key=tok.key,
                            value=new,
                            body_offset=tok.body_offset,
                            body_length=len(new_text),
                        )
                        _shift_cached_param_tokens(
                            self._param_tokens,
                            edit_old_end=old_body_end,
                            delta=delta,
                            exclude_key=existing,
                        )
                        self._model_token = _shift_single_token(
                            self._model_token, old_body_end, delta
                        )
                        return
                    except ValueError:
                        pass
                self._canonical_rerender()
                return
        self.params[key] = new
        self._canonical_rerender()

    def remove_param(self, key: str) -> None:
        """Remove a parameter (no-op if absent)."""
        for existing in list(self.params):
            if existing.lower() == key.lower():
                del self.params[existing]
                self._param_tokens.pop(existing, None)
                self._canonical_rerender()
                return

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
                _shift_cached_param_tokens(
                    self._param_tokens,
                    edit_old_end=old_body_end,
                    delta=delta,
                )
                return
            except ValueError:
                pass
        self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        parts = [self.ref, *self.nodes]
        if self.model is not None:
            parts.append(self.model)
            # A model-kind element may carry a trailing value (diode area, switch
            # ON/OFF state) after the model name; preserve it on re-render.
            if self.value is not None:
                parts.append(self.value)
        elif self.value is not None and self._value_param_key is None:
            parts.append(self.value)
        for k, v in self.params.items():
            parts.append(f"{k}={v}")
        body = " ".join(parts)
        self.card.replace_body(body)
        self._model_token = None
        self._param_tokens.clear()

    def display_value(self) -> str:
        """Single-string projection for display: model name for M/Q/J/X,
        value field for R/C/L/V/I/E/F/G/H, KV pair for B-sources, empty
        for malformed bodies."""
        if self.model is not None:
            return self.model
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
    truncated value."""
    try:
        toks = tokenize_body(body)
    except Exception:
        return False
    seen_kv = False
    for tok in toks:
        if tok.kind == TokenKind.KEY_VALUE:
            seen_kv = True
            continue
        if seen_kv and tok.kind in (TokenKind.PARENED, TokenKind.BARE):
            return True
    return False


def instances_by_ref(cards) -> dict[str, SpiceCard]:
    """Lowercased ``ref`` → instance-card lookup. Skips cards with no ref."""
    return {c.name.lower(): c for c in cards if c.kind == "instance" and c.name}


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
    param_defaults: dict[str, str] = field(default_factory=dict)
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
        defaults: dict[str, str] = {}
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
                    assert tok.key is not None
                    assert tok.value is not None
                    defaults[tok.key] = tok.value
            else:
                if tok.kind == TokenKind.BARE:
                    ports.append(tok.text)
                elif tok.kind == TokenKind.KEY_VALUE:
                    # ``.SUBCKT NAME P1 K=V`` — implicit PARAMS:; many
                    # SPICE flavours accept this without the marker.
                    in_params = True
                    assert tok.key is not None
                    assert tok.value is not None
                    defaults[tok.key] = tok.value
        return cls(
            card=card,
            name=name,
            ports=ports,
            param_defaults=defaults,
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

    def set_param_default(self, key: str, value: float | str) -> None:
        """Set or add a parameter default. Canonical rerender."""
        new = _format_value(value)
        for existing in list(self.param_defaults):
            if existing.lower() == key.lower():
                self.param_defaults[existing] = new
                self._canonical_rerender()
                return
        self.param_defaults[key] = new
        self._canonical_rerender()

    def remove_param_default(self, key: str) -> None:
        """Remove a parameter default (no-op if absent)."""
        for existing in list(self.param_defaults):
            if existing.lower() == key.lower():
                del self.param_defaults[existing]
                self._canonical_rerender()
                return

    def _canonical_rerender(self) -> None:
        parts: list[str] = [".SUBCKT", self.name, *self.ports]
        if self.param_defaults:
            if self._used_params_marker:
                parts.append("PARAMS:")
            for k, v in self.param_defaults.items():
                parts.append(f"{k}={v}")
        body = " ".join(parts)
        self.card.replace_body(body)
        self._name_token = None


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
    pre-extracted ``function_calls`` and ``signal_refs`` so callers
    (esp. ``spice_validator``) can run semantic checks without
    re-parsing characters.

    Setters operate on identity fields (analysis, label) only — the
    measurement body itself is not mutated through this view.
    """

    card: SpiceCard
    analysis: MeasAnalysis | None
    name: str
    body_tokens: list[Token]
    function_calls: list[FunctionCall]
    signal_refs: list[str]
    _directive_text: str = ".MEAS"  # ``.MEAS`` or ``.MEASURE``

    @classmethod
    def from_card(cls, card: SpiceCard) -> MeasCard:
        _require_kind(card, "meas")
        tokens = tokenize_body(card.body)
        # tokens[0] = ".MEAS" (or ".MEASURE"). tokens[1] is either
        # analysis kind or measurement label.
        if len(tokens) < 2:
            raise _malformed(card, "malformed .MEAS")
        directive = tokens[0].text
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
        signal_refs: list[str] = []
        _collect_function_calls(body_tokens, function_calls, signal_refs)

        return cls(
            card=card,
            analysis=analysis,
            name=meas_name,
            body_tokens=body_tokens,
            function_calls=function_calls,
            signal_refs=signal_refs,
            _directive_text=directive,
        )

    def set_label(self, new_name: str) -> None:
        """Rename the measurement label. Canonical rerender."""
        self.name = new_name
        self._canonical_rerender()

    def set_analysis(self, kind: MeasAnalysis | None) -> None:
        """Set or clear the analysis-kind keyword. Canonical rerender."""
        self.analysis = kind
        self._canonical_rerender()

    def _canonical_rerender(self) -> None:
        parts: list[str] = [self._directive_text]
        if self.analysis is not None:
            parts.append(self.analysis.upper())
        parts.append(self.name)
        for tok in self.body_tokens:
            if tok.kind == TokenKind.COMMENT_TRAIL:
                break
            parts.append(tok.text)
        body = " ".join(parts)
        self.card.replace_body(body)


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


def _shift_cached_param_tokens(
    cache: dict[str, Token],
    *,
    edit_old_end: int,
    delta: int,
    exclude_key: str | None = None,
) -> None:
    """Shift body_offset on cache entries whose original position sat past the edit.

    Called by view setters after a successful ``replace_span`` so any
    *other* cached tokens stay aligned with the mutated body. ``delta``
    is the length change (new minus old span). Tokens with
    ``body_offset >= edit_old_end`` shift by ``delta``; tokens before
    the edit are unchanged. ``exclude_key`` is the token that was just
    edited and re-cached at its (unchanged) ``body_offset``.
    """
    if delta == 0:
        return
    for k, tok in list(cache.items()):
        if k == exclude_key:
            continue
        if tok.body_offset >= edit_old_end:
            cache[k] = Token(
                kind=tok.kind,
                text=tok.text,
                key=tok.key,
                value=tok.value,
                body_offset=tok.body_offset + delta,
                body_length=tok.body_length,
            )


def _shift_single_token(tok: Token | None, edit_old_end: int, delta: int) -> Token | None:
    """Same as ``_shift_cached_param_tokens`` for a single optional token."""
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


def _collect_function_calls(
    tokens: Sequence[Token],
    out_calls: list[FunctionCall],
    out_signal_refs: list[str],
) -> None:
    """Walk ``tokens`` and append any ``BARE PARENED`` function calls.

    Recurses into the contents of each PARENED token so nested calls
    are surfaced too (``mag(V(out))`` yields both ``mag`` and ``V``).
    Also recurses into BRACED contents — ``{vdb(out)}`` should still
    surface ``vdb`` for validator rules.
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < n else None
        if t.kind == TokenKind.BARE and nxt is not None and nxt.kind == TokenKind.PARENED:
            args = nxt.text[1:-1]
            out_calls.append(FunctionCall(name=t.text, args_text=args))
            if t.text.lower() in ("v", "i"):
                out_signal_refs.append(args.strip())
        # Recurse into PARENED / BRACED contents.
        if t.kind in (TokenKind.PARENED, TokenKind.BRACED):
            inner_text = t.text[1:-1]
            try:
                inner_tokens = tokenize_body(inner_text)
            except SpiceLexError:
                continue
            _collect_function_calls(inner_tokens, out_calls, out_signal_refs)
