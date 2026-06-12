"""Pre-flight validation for SPICE directives.

The simulator's measurement engine has a narrower expression vocabulary
than the waveform viewer. Directives that look syntactically correct
(e.g. ``.MEAS AC fc WHEN vdb(out)=-3``) silently fail post-hoc inside
the .log file. This module catches the common pitfalls before the
directive is written to disk.

Layer A is intentionally a narrow blocklist of patterns we've actually
hit, not an exhaustive grammar. It grows as we find more.

Implementation note: rules walk classified tokens (``MeasCard.function_calls``)
rather than substring-matching regex. This keeps cases like
``.MEAS WHEN x=vdb_safe`` (a variable name that happens to start with
``vdb``) from being false-flagged, and opens the door to checks the
regex couldn't do — signal-reference resolution, analysis-kind
mismatches, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import ELEMENT_SPECS, InstanceLine, MeasCard, SubcktCard

# Analysis directives that LTspice accepts. Used by validate_netlist to
# match ``.meas <kind>`` against the active analysis and detect duplicates.
# Note that ``.meas noise`` is not a valid form even though ``.noise`` is —
# the meas-side set is intentionally narrower.
ANALYSIS_KINDS: frozenset[str] = frozenset({"tran", "ac", "dc", "op", "noise"})
MEAS_KINDS: frozenset[str] = frozenset({"tran", "ac", "dc", "op"})
# ``.op`` is a bias-point request, not a mutually-exclusive analysis: LTspice
# runs it alongside exactly one real analysis (verified live: ``.op``+``.tran``
# and ``.op``+``.ac`` run; ``.ac``+``.tran`` is rejected). The "more than one
# analysis" gate counts only these, so it doesn't false-positive on the common
# ``.op`` + analysis idiom. ``.op`` stays in ANALYSIS_KINDS/MEAS_KINDS
# for ``.meas op`` matching.
EXCLUSIVE_ANALYSIS_KINDS: frozenset[str] = ANALYSIS_KINDS - frozenset({"op"})

AnalysisKind = Literal["tran", "ac", "dc", "op", "noise"]


@dataclass(frozen=True)
class ValidationError:
    """Result of a failed validation rule."""

    rule_name: str
    message: str
    suggestion: str


@dataclass(frozen=True)
class _Rule:
    name: str
    blocked_function: str  # case-insensitive match against MeasCard function calls
    simulators: frozenset[str]  # empty => all simulators
    message: str
    suggestion: str


_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="vdb_in_meas",
        # LTspice's measurement engine accepts mag/re/im/ph but NOT vdb —
        # that's a waveform-viewer-only function.
        blocked_function="vdb",
        simulators=frozenset({"LTspice"}),
        message=(
            "vdb() is not accepted in .MEAS directives — it's a "
            "waveform-viewer-only function in LTspice. .MEAS only "
            "supports mag(), re(), im(), ph()."
        ),
        suggestion=(
            "Use mag(V(node)) and convert to dB downstream, or rely on "
            "bode_metrics(mode='filter') for −3 dB cutoffs and "
            "bode_metrics(mode='point') for point queries."
        ),
    ),
    _Rule(
        name="phase_in_meas",
        blocked_function="phase",
        simulators=frozenset({"LTspice"}),
        message=(
            "phase() is not accepted in .MEAS directives — it's a "
            "waveform-viewer-only function. Use ph() instead."
        ),
        suggestion=("Replace phase(...) with ph(...) — ph() is the .MEAS-compatible spelling."),
    ),
    _Rule(
        name="group_delay_in_meas",
        blocked_function="group_delay",
        simulators=frozenset({"LTspice"}),
        message=(
            "group_delay() is not accepted in .MEAS directives — it's a "
            "waveform-viewer-only function in LTspice."
        ),
        suggestion=(
            "Compute group delay via bode_metrics(mode='point') with "
            "include_unwrapped_phase=True, then numerically differentiate "
            "the unwrapped phase."
        ),
    ),
)


def validate_directive(directive: str, simulator: str = "LTspice") -> ValidationError | None:
    """Check a directive against the blocklist for the given simulator.

    Returns the first matched rule's error, or None if no rule fires.
    Empty / whitespace-only input is a no-op. Non-``.MEAS`` directives
    pass through unchecked (no current rule applies to them).
    """
    if not directive:
        return None
    stripped = directive.lstrip()
    if not stripped:
        return None
    if not stripped.lower().startswith(".meas"):
        return None

    # Parse via spice_lex to get classified tokens.
    text = directive if directive.endswith("\n") else directive + "\n"
    try:
        cards = lex(text).cards
    except SpiceLexError:
        # Tokenizer faults aren't a validator concern; let downstream
        # catch them. Substring-fallback is intentionally not done — a
        # broken directive shouldn't be silently approved.
        return None
    meas_cards = [c for c in cards if c.kind == "meas"]
    if not meas_cards:
        return None
    try:
        meas = MeasCard.from_card(meas_cards[0])
    except SpiceLexError:
        return None

    called = {fc.name.lower() for fc in meas.function_calls}
    for rule in _RULES:
        if rule.simulators and simulator not in rule.simulators:
            continue
        if rule.blocked_function.lower() in called:
            return ValidationError(
                rule_name=rule.name,
                message=rule.message,
                suggestion=rule.suggestion,
            )
    return None


def _card_directive(card: SpiceCard) -> str:
    """Source-form display text of a card for issue payloads."""
    return card.raw_lines[0].rstrip() if card.raw_lines else card.body.strip()


def validate_netlist_arity(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag instance cards whose positional-node count is below the per-
    element minimum, and B-sources missing the ``V=``/``I=`` prefix.

    Returns dicts matching the existing ``handle_validate_netlist`` issue
    shape: ``{line, directive, message, suggestion}``. These are bodies
    LTspice rejects at simulation time with ``Expected 2 node names
    here`` or ``Unknown parameter``.
    """
    issues: list[dict[str, object]] = []
    for card in cards:
        if card.kind != "instance":
            continue
        try:
            inst = InstanceLine.from_card(card)
        except SpiceLexError:
            continue
        prefix = inst.ref[:1].upper()
        spec = ELEMENT_SPECS.get(prefix)
        if spec is None:
            continue

        directive = _card_directive(card)

        # E/G/F/H carry two body shapes:
        #   - positional gain (``E1 out 0 in 0 10``) — needs spec.min_nodes
        #     positional tokens (4 for E/G, 2 for F/H).
        #   - keyed behavioral (``E1 out 0 VALUE={...}`` / ``POLY(...)`` /
        #     ``TABLE(...)``) — the keyed form moves the value into params
        #     and only the two output nodes remain positional.
        # spec.kind_for(has_kv) distinguishes the forms via the
        # kv_means_params_only flag, so the threshold collapses to "2
        # output nodes" whenever KV is present on a keyed-capable prefix.
        has_kv = bool(inst.params)
        required = 2 if spec.kind_for(has_kv=has_kv) == "params_only" else spec.min_nodes

        # Some SPICE dialects accept keyed primary-value forms
        # (``R1 a b R=1k`` / ``C1 a b C=1n`` / ``L1 a b L=1u``), where all
        # positional tokens after the ref are nodes. Re-count from the raw
        # token stream when such a primary-value KV is present so arity checks
        # do not confuse the last node with a positional value.
        node_count = len(inst.nodes)
        primary_value_key = next(
            (key for key in inst.params if key.upper() == prefix),
            None,
        )
        if prefix in ("R", "C", "L") and primary_value_key is not None:
            positionals_after_ref = sum(
                1
                for tok in tokenize_body(card.body)[1:]
                if tok.kind not in (TokenKind.KEY_VALUE, TokenKind.COMMENT_TRAIL)
            )
            node_count = positionals_after_ref
            # Real LTspice 26 accepts R=<value>, but rejects C=<value> and
            # L=<value> as unknown parameters. ngspice accepts all three; this
            # validator is the LTspice pre-flight path.
            if prefix in ("C", "L"):
                rewrite = " ".join([inst.ref, *inst.nodes, inst.value or ""])
                issues.append(
                    {
                        "line": card.line_start,
                        "directive": directive,
                        "message": (
                            f"{inst.ref}: LTspice does not accept {prefix}= as "
                            f"the primary value for a {prefix}-element; use the "
                            "positional value form instead."
                        ),
                        "suggestion": f"Rewrite as `{rewrite}`.",
                    }
                )

        if node_count < required:
            issues.append(
                {
                    "line": card.line_start,
                    "directive": directive,
                    "message": (
                        f"{inst.ref}: expected at least {required} "
                        f"positional node(s) for a {prefix}-element, got {node_count}"
                    ),
                }
            )

        if prefix == "B":
            kv_keys = {k.upper() for k in inst.params}
            if "V" not in kv_keys and "I" not in kv_keys:
                issues.append(
                    {
                        "line": card.line_start,
                        "directive": directive,
                        "message": (
                            f"{inst.ref}: B-source requires V= or I= prefix on the expression "
                            f"(got params {sorted(inst.params)})"
                        ),
                    }
                )

    return issues


# Node spellings that are always considered connected (the SPICE ground).
_GROUND_NODES = frozenset({"0", "gnd"})

# Characters that cannot appear in a plain node name. Positional tokens
# carrying them are value syntax that spilled into the node slots (PULSE
# fragments, braced expressions), not nodes.
_NON_NODE_CHARS = frozenset("(){}=\"'")

# How many leading positional tokens of an instance card are element
# terminals, per prefix. Deliberately lint-local: ELEMENT_SPECS drives the
# arity *errors* and stays narrow, while the dangling *warning* must know
# every standard prefix — a prefix absent here would zero out its nodes'
# counts and fabricate warnings on neighbouring elements.
_TERMINAL_COUNTS: dict[str, int] = {
    "R": 2,
    "C": 2,
    "L": 2,
    "V": 2,
    "I": 2,
    "B": 2,
    "D": 2,
    "F": 2,  # out+ out- (then controlling-source ref, gain)
    "H": 2,
    "W": 2,  # n+ n- (then controlling-source ref, model)
    "Q": 3,  # c b e (optional substrate suppressed, not counted)
    "J": 3,
    "Z": 3,
    "U": 3,
    "M": 4,  # d g s b
    "S": 4,  # two switch nodes + two control nodes
    "T": 4,
    "O": 4,
    "K": 0,  # positionals are inductor refs, not nodes
}

# Prefixes whose final positional token is a model name, stripped before
# the terminal slice so a short form (e.g. a 3-node MOSFET) cannot count
# its model name as a terminal.
_TRAILING_MODEL_PREFIXES = frozenset({"D", "J", "M", "O", "Q", "S", "U", "W", "Z"})

# V(...) / I(...) probe references inside expression text on instance
# cards (B/E/G bodies, behavioural params). Probed identifiers are
# connections too — they feed the suppressor set, never a terminal count.
_PROBE_REF_RE = re.compile(r"\b[VI]\s*\(([^()]*)\)", re.IGNORECASE)


def _instance_terminals(inst: InstanceLine) -> tuple[list[str], list[str]]:
    """Split an instance view into ``(terminal tokens, other positionals)``.

    Terminals are the leading token positions known to carry circuit nodes
    for the element's prefix. Every other positional — model names, area
    factors, controlling-source refs, POLY tails, the X subckt name, all
    tokens of an unknown prefix (XSPICE A-cards) — lands in the second slot:
    those feed the suppressor set, so a card shape this table mis-models
    can only ever hide a warning, never invent one.
    """
    prefix = inst.ref[:1].upper()

    if prefix == "X":
        # Every positional up to the subckt name is a node.
        return list(inst.nodes), [inst.model] if inst.model else []

    candidates = list(inst.nodes)
    tail: list[str] = []
    if inst.model is not None:
        if prefix in _TRAILING_MODEL_PREFIXES:
            tail.append(inst.model)
        else:
            # Prefixes outside ELEMENT_SPECS parse with the default spec,
            # which reads the last positional as a model name even when it
            # is really a node (the T-line's 4th port) — keep it positional
            # so the terminal slice can count it.
            candidates.append(inst.model)
    elif inst.value is not None:
        # The keyed primary-value form (``R1 a b R=1k``) reads its value
        # from the KEY=VALUE token, not a positional slot — only a
        # positional value rejoins the candidates.
        keyed = prefix in ("R", "C", "L") and any(k.upper() == prefix for k in inst.params)
        if not keyed:
            candidates.append(inst.value)

    if prefix in ("E", "G"):
        # The plain gain form (out+ out- in+ in- gain) carries 4 node
        # slots; POLY / VALUE= / truncated shapes guarantee only the two
        # output nodes. ``value`` is set only on the KV-free positional
        # form, so it discriminates without re-reading raw tokens.
        count = 4 if inst.value is not None and len(inst.nodes) == 4 else 2
        return candidates[:count], candidates[count:]
    count = _TERMINAL_COUNTS.get(prefix)
    if count is None:
        return [], candidates + tail
    return candidates[:count], candidates[count:] + tail


@dataclass
class _NodeUse:
    """First reference to a node within one scope, plus total reference count.

    Only the first occurrence's details are ever reported, so later
    references just bump ``count``.
    """

    display: str
    referrer: str
    line: int
    directive: str
    is_port: bool
    count: int = 1


def drop_title_card(cards: list[SpiceCard]) -> list[SpiceCard]:
    """Drop the line-1 instance card produced by a SPICE deck's title.

    Line 1 of a netlist is a free-text title by SPICE convention. The
    lexer has no title concept, so a title starting with an element
    letter parses as an instance card — its words would otherwise be
    counted as circuit nodes by instance-level lints.
    """
    return [card for card in cards if not (card.kind == "instance" and card.line_start == 1)]


def validate_netlist_dangling_nodes(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag nodes referenced by exactly one element terminal in their scope.

    Warning-level companion to ``validate_netlist_arity`` — the caller
    attaches severity. A single-connection node is legal SPICE (bias
    fragments and test stubs leave nodes open on purpose), so each issue
    only states the fact and names the one referencing element.

    Occurrences are counted per scope: top level and each ``.SUBCKT``
    body separately, with the header's port names counting once inside
    the body (a port wired to one body element is fully connected).
    Ground (``0`` / ``gnd``) and ``.GLOBAL`` nodes are excluded.

    Counting is asymmetric by design: only token positions known to be
    element terminals are counted, while every other positional token and
    every identifier probed via ``V(...)``/``I(...)`` in instance-card
    expressions joins a suppressor set that vetoes the warning. A token
    the lint cannot classify can therefore only hide a warning, never
    create a false one.
    """
    global_nodes: set[str] = set()
    for card in cards:
        if card.kind != "directive":
            continue
        if not card.body.lstrip().lower().startswith(".global"):
            continue
        try:
            tokens = tokenize_body(card.body)
        except SpiceLexError:
            continue
        if not tokens or tokens[0].text.lower() != ".global":
            continue
        for tok in tokens[1:]:
            if tok.kind == TokenKind.COMMENT_TRAIL:
                break
            if tok.kind == TokenKind.BARE:
                global_nodes.add(tok.text.lower())

    # scope → lowercased node → first reference + total reference count.
    occurrences: dict[tuple[str, ...], dict[str, _NodeUse]] = {}
    suppressed: set[str] = set()

    def record(
        scope: tuple[str, ...], node: str, referrer: str, card: SpiceCard, is_port: bool = False
    ) -> None:
        key = node.lower()
        if key in _GROUND_NODES or key in global_nodes:
            return
        nodes = occurrences.setdefault(scope, {})
        use = nodes.get(key)
        if use is None:
            nodes[key] = _NodeUse(
                display=node,
                referrer=referrer,
                line=card.line_start,
                directive=_card_directive(card),
                is_port=is_port,
            )
        else:
            use.count += 1

    for card in cards:
        if card.kind == "subckt":
            try:
                sub = SubcktCard.from_card(card)
            except SpiceLexError:
                continue
            body_scope = (*card.scope, sub.name)
            for port in sub.ports:
                record(body_scope, port, f".SUBCKT {sub.name}", card, is_port=True)
        elif card.kind == "instance":
            try:
                inst = InstanceLine.from_card(card)
            except SpiceLexError:
                continue
            terminals, rest = _instance_terminals(inst)
            for node in terminals:
                if set(node) & _NON_NODE_CHARS:
                    continue
                record(card.scope, node, inst.ref, card)
            suppressed.update(tok.lower() for tok in rest)
            for probe in _PROBE_REF_RE.finditer(card.body):
                for part in probe.group(1).split(","):
                    name = part.strip()
                    if name:
                        suppressed.add(name.lower())

    issues: list[dict[str, object]] = []
    for scope, nodes in occurrences.items():
        where = f" inside .SUBCKT {scope[-1]}" if scope else ""
        for key, use in nodes.items():
            if use.count != 1 or key in suppressed:
                continue
            if use.is_port:
                message = (
                    f"Node '{use.display}' is declared as a port of .SUBCKT "
                    f"{scope[-1]} but connected to no element terminal in its body."
                )
                suggestion = (
                    "Wire the port to an element inside the body — or drop it "
                    "from the .SUBCKT header if it is unused."
                )
            else:
                message = (
                    f"Node '{use.display}' is connected to only one element "
                    f"terminal ({use.referrer}){where}."
                )
                suggestion = (
                    "Wire the node to a second terminal — or ignore this "
                    "warning if the netlist is a deliberately unterminated fragment."
                )
            issues.append(
                {
                    "line": use.line,
                    "directive": use.directive,
                    "message": message,
                    "suggestion": suggestion,
                }
            )
    return issues


def list_rules() -> list[dict[str, object]]:
    """Return the active rule set, useful for tool-side error suggestions."""
    return [
        {
            "name": r.name,
            "simulators": sorted(r.simulators) if r.simulators else ["*"],
            "message": r.message,
            "suggestion": r.suggestion,
        }
        for r in _RULES
    ]
