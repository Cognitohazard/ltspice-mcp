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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import (
    ELEMENT_SPECS,
    InstanceLine,
    MeasCard,
    SubcktCard,
    body_has_stray_kv_remnant,
)

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


def _validate_tran_tstep(directive: str, simulator: str) -> ValidationError | None:
    """Flag ``.tran 0 <tstop>`` (zero step time) when targeting ngspice.

    LTspice accepts a zero step time as a request to pick the timestep itself
    (auto-timestep). ngspice requires a positive step time and aborts the run
    with "TSTEP is invalid". Only fires for the ngspice target so a clean
    LTspice deck stays clean. The directive form is
    ``.tran <Tstep> <Tstop> [<Tstart> [<Tmax>]] [modifiers]``; a parameterised
    or non-numeric Tstep/Tstop is left alone (resolved at run time).
    """
    if simulator != "ngspice":
        return None
    tokens = directive.split()[1:]
    if len(tokens) < 2:
        # A lone value is a bare Tstop (an LTspice shorthand), not Tstep=0.
        return None
    try:
        tstep = parse_spice_value(tokens[0])
        tstop = parse_spice_value(tokens[1])
    except ValueError:
        return None
    if tstep != 0 or tstop <= 0:
        return None
    return ValidationError(
        rule_name="tran_tstep_zero_ngspice",
        message=(
            "ngspice rejects a zero step time (.tran 0 ...) with 'TSTEP is "
            "invalid'; LTspice accepts it as auto-timestep. This deck will fail "
            "on ngspice."
        ),
        suggestion=(
            "Give a positive step time, e.g. '.tran <tstep> <tstop>' with "
            "tstep around tstop/1000, or run it on LTspice."
        ),
    )


def validate_directive(directive: str, simulator: str = "LTspice") -> ValidationError | None:
    """Check a directive against the pre-flight rules for the given simulator.

    Returns the first matched rule's error, or None if no rule fires.
    Empty / whitespace-only input is a no-op. Covers ``.MEAS`` function
    blocklists and the ``.tran`` zero-step ngspice incompatibility; other
    directives pass through unchecked.
    """
    if not directive:
        return None
    stripped = directive.lstrip()
    if not stripped:
        return None
    lower = stripped.lower()
    if lower.startswith(".tran"):
        return _validate_tran_tstep(stripped, simulator)
    if not lower.startswith(".meas"):
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


def validate_netlist_arity(
    cards: list[SpiceCard], *, simulator: str = "LTspice"
) -> list[dict[str, object]]:
    """Flag instance cards whose positional-node count is below the per-
    element minimum, and B-sources missing the ``V=``/``I=`` prefix.

    Returns dicts matching the existing ``handle_validate_netlist`` issue
    shape: ``{line, directive, message, suggestion}``. The node-count and
    B-source checks are simulator-agnostic. ``simulator`` gates the one
    LTspice-only rule (the ``C=``/``L=`` primary-value rejection): under a
    non-LTspice target it is suppressed, since ngspice accepts those forms.
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
            # L=<value> as unknown parameters. ngspice accepts all three, so
            # this rejection only applies to the LTspice target. (The
            # node-count recompute above stays for both — it keeps the keyed
            # value from being miscounted as a node either way.)
            if prefix in ("C", "L") and simulator == "LTspice":
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

        # A behavioral expression with whitespace around its operators
        # (``V = V(a) + V(b)``) leaves orphan tokens after the first key=value:
        # the lexer re-joins only glued forms (``V=V(a)+V(b)``). Flag the
        # remnant so it is not silently dropped on read or corrupted by a
        # value-edit that only rewrites the first key=value span.
        if body_has_stray_kv_remnant(card.body):
            issues.append(
                {
                    "line": card.line_start,
                    "directive": directive,
                    "message": (
                        f"{inst.ref}: value expression is not fully parsed — "
                        "tokens after the first key=value are orphaned. Wrap the "
                        "expression in braces (e.g. V={...}) so it reads and "
                        "edits as a single value."
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


def _scan_global_nodes(cards: list[SpiceCard]) -> set[str]:
    """Collect lowercased node names declared via ``.GLOBAL`` directives.

    Shared by the dangling-node and bias-topology passes — a ``.GLOBAL``
    node is connected across every scope by convention, so neither pass
    should treat it as locally unterminated or floating.
    """
    nodes: set[str] = set()
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
                nodes.add(tok.text.lower())
    return nodes


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
    global_nodes = _scan_global_nodes(cards)

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


# A clean node/device identifier inside V(...)/I(...): letters, digits,
# underscore, and the LTspice '!' net suffix. Anything else (operators,
# braces, a ':' or '.' hierarchy separator) marks an expression fragment or a
# hierarchical reference we deliberately don't adjudicate.
_PLAIN_REF_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_!]*")


def _known_names(cards: list[SpiceCard]) -> set[str]:
    """Over-approximate the set of names a directive may reference.

    Every positional token of every instance card (terminals, model names,
    values, the ref itself), every ``.SUBCKT`` port, ``.GLOBAL`` node, and
    ground — all lowercased. Deliberately a superset: a name appearing
    anywhere at element level is treated as defined, so the dangling-reference
    pass can only fire on a name that is genuinely absent, never on one it
    failed to classify.
    """
    known: set[str] = set(_GROUND_NODES)
    known |= _scan_global_nodes(cards)
    for card in cards:
        if card.kind == "subckt":
            try:
                sub = SubcktCard.from_card(card)
            except SpiceLexError:
                continue
            known.add(sub.name.lower())
            known.update(p.lower() for p in sub.ports)
        elif card.kind == "instance":
            try:
                inst = InstanceLine.from_card(card)
            except SpiceLexError:
                continue
            known.add(inst.ref.lower())
            terminals, rest = _instance_terminals(inst)
            known.update(t.lower() for t in terminals)
            known.update(r.lower() for r in rest)
    return known


def validate_netlist_directive_refs(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag ``V(...)``/``I(...)`` references to names that exist nowhere.

    A ``.meas``, output directive, or behavioral source that names a node no
    element connects to — the classic case being a schematic net wired but
    never labeled, so it exports as ``N00x`` while the directive still asks for
    ``V(vref)`` and silently resolves to nothing. Matching is
    case-insensitive; hierarchical refs (``V(X1:out)``) and expression
    fragments (``V(a*2)``) are left alone. Each genuinely-missing name is
    reported once. The caller attaches severity.
    """
    cards = drop_title_card(cards)
    known = _known_names(cards)

    issues: list[dict[str, object]] = []
    seen: set[str] = set()
    for card in cards:
        # .meas is its own kind; .print/.plot/.save/.four are "directive";
        # B/E/G behavioral sources are "instance" — all can name nodes.
        if card.kind not in ("instance", "directive", "meas"):
            continue
        # .func defines a formal-parameter expression; its V(formal) is not a node.
        if card.kind == "directive" and card.body.lstrip().lower().startswith(".func"):
            continue
        for probe in _PROBE_REF_RE.finditer(card.body):
            kind = probe.group(0)[0].upper()  # 'V' or 'I'
            for part in probe.group(1).split(","):
                name = part.strip()
                if not name or not _PLAIN_REF_RE.fullmatch(name):
                    continue
                key = name.lower()
                if key in known or key in seen:
                    continue
                seen.add(key)
                issues.append(
                    {
                        "line": card.line_start,
                        "directive": _card_directive(card),
                        "message": (
                            f"References {kind}({name}) but no node or device named "
                            f"'{name}' exists in the netlist."
                        ),
                        "suggestion": (
                            "Check the name for a typo; if it is a schematic net that "
                            "was wired but not labeled, add a net label so the name "
                            "survives export (an unlabeled net becomes N00x)."
                        ),
                    }
                )
    return issues


class _DSU:
    """Minimal union-find over node names for DC-reachability grouping."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        parent = self._parent
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# Sentinel node that every ground / sink unions into, so "has a DC path to
# ground" reduces to "shares a DSU root with this". A NUL byte can never
# occur in a real node name, so the sentinel cannot collide with one.
_GROUND_SINK = "\x00ground\x00"

# Element prefixes whose terminals carry NO DC current path: a capacitor's
# dielectric, a current source's branch, and the current outputs of a VCCS
# (G) / CCCS (F). These are the only provable DC opens — every other element
# gets a conservative conductive edge (see ``_dc_edges``).
_NO_DC_PATH_PREFIXES = frozenset({"C", "I", "G", "F"})

# Terminal slot of a MOSFET's insulated gate (``M nd ng ns nb``): a true DC
# open that conducts to nothing, and the node a floating-gate message names.
_MOSFET_GATE_INDEX = 1


def _star(nodes: list[str]) -> list[tuple[str, str]]:
    """Edges that union ``nodes`` into one component (star from the first)."""
    return [(nodes[0], n) for n in nodes[1:]]


def _b_source_is_dc_open(inst: InstanceLine) -> bool:
    """True for a behavioral ``B`` source that carries no DC path between its
    nodes: a current source (``I=``) with no parallel resistor (``Rpar``).

    ngspice's arbitrary source is a current source when ``I=`` is given and a
    voltage source when ``V=`` is given. LTspice additionally allows ``Rpar``
    (an explicit parallel resistor — the documented fix for a floating
    current-source node), which restores a DC path. ``Cpar`` is a parallel
    *capacitor*, open at DC, so it does not. Anything else (``V=``, a
    behavioral ``R=``, a power source) conducts and keeps the default edge.
    """
    keys = {k.lower() for k in inst.params}
    return "i" in keys and "v" not in keys and "rpar" not in keys


def _dc_edges(
    prefix: str, terminals: list[str], *, b_dc_open: bool = False
) -> list[tuple[str, str]]:
    """Node pairs an element makes DC-conductive, for the bias-topology graph.

    Conservative over-connection: an element contributes edges wherever DC
    conduction is *possible*. Only provable opens contribute none — a
    capacitor's dielectric, a MOSFET's gate oxide, current-source /
    current-output branches, and a behavioral current source (``b_dc_open``).
    A missing edge can only hide a no-DC-path warning, never invent one, so
    bias-dependent devices (diodes, transistor channels, switches) are all
    treated as conductive.
    """
    if prefix in _NO_DC_PATH_PREFIXES:
        return []
    if prefix == "B" and b_dc_open:
        # Behavioral CURRENT source (I=, no Rpar): pins current, not voltage —
        # the same DC open as the independent I element.
        return []
    if prefix in ("E", "H"):
        # Controlled VOLTAGE source: only the output pair is pinned; the
        # controlling inputs are high-impedance sense nodes.
        return _star(terminals[:2])
    if prefix == "M":
        # The insulated gate is a true DC open; the drain, source, and bulk
        # terminals form one conductive group.
        return _star([t for i, t in enumerate(terminals) if i != _MOSFET_GATE_INDEX])
    return _star(terminals)


def _apply_instance_edges(
    dsu: _DSU,
    inst: InstanceLine,
    terminals: list[str],
    subckt_grounds: dict[str, frozenset[int]],
) -> list[str]:
    """Union an instance's DC-conductive terminal pairs into ``dsu``.

    ``terminals`` may be name-keyed (per-scope graph) or scope-qualified
    (deck-wide global graph); the source kind comes from ``inst``. Returns
    the external nodes an ``X`` instance is grounded through — the ports its
    subckt body ties to ground (node ``0`` / ``.GLOBAL``) internally, which
    the black-box clique alone cannot see. The caller folds those into its
    ground set.
    """
    prefix = inst.ref[:1].upper()
    b_dc_open = prefix == "B" and _b_source_is_dc_open(inst)
    for a, b in _dc_edges(prefix, terminals, b_dc_open=b_dc_open):
        if _NON_NODE_CHARS.isdisjoint(a) and _NON_NODE_CHARS.isdisjoint(b):
            dsu.union(a.lower(), b.lower())
    if prefix == "X" and inst.model:
        nested = subckt_grounds.get(inst.model.lower(), frozenset())
        return [terminals[i].lower() for i in nested if i < len(terminals)]
    return []


def _iter_instances(
    cards: list[SpiceCard],
) -> Iterator[tuple[SpiceCard, InstanceLine, list[str]]]:
    """Yield ``(card, view, terminals)`` for each instance card that lexes,
    skipping the ones the view rejects. Shared by every DC-graph walk."""
    for card in cards:
        if card.kind != "instance":
            continue
        try:
            inst = InstanceLine.from_card(card)
        except SpiceLexError:
            continue
        yield card, inst, _instance_terminals(inst)[0]


def _union_instance(
    dsu: _DSU,
    inst: InstanceLine,
    terminals: list[str],
    subckt_grounds: dict[str, frozenset[int]],
) -> None:
    """Apply an instance's DC edges and fold its ``X``-internal grounds into
    the ground sentinel — the ground-reachability walks' inner step."""
    for node in _apply_instance_edges(dsu, inst, terminals, subckt_grounds):
        dsu.union(_GROUND_SINK, node)


@dataclass
class _BiasGraph:
    """Per-scope DC-reachability accumulator for the bias-topology pass."""

    dsu: _DSU = field(default_factory=_DSU)
    display: dict[str, str] = field(default_factory=dict)
    first_card: dict[str, SpiceCard] = field(default_factory=dict)
    # node -> the refdes of every terminal that references it (the source
    # prefix is recovered as ref[0]); its length is the node's degree.
    referrers: dict[str, list[str]] = field(default_factory=dict)
    gate_of: dict[str, str] = field(default_factory=dict)
    sinks: set[str] = field(default_factory=set)
    # Every element's full terminal set (capacitors included), so a
    # physically-contiguous but internally-AC-coupled floating region can be
    # collapsed to one issue — the DC graph alone cannot represent it.
    phys_groups: list[list[str]] = field(default_factory=list)


def _subckt_internal_grounds(
    cards: list[SpiceCard], global_nodes: set[str]
) -> dict[str, frozenset[int]]:
    """Map each ``.SUBCKT`` name to the indices of its ports that reach
    ground (node ``0`` / ``.GLOBAL``) through the body's own elements.

    Node ``0`` is global, so a subckt can reference ground internally
    without a dedicated ground port — op-amp / regulator / sensor
    macromodels routinely do. An ``X`` instance of such a subckt biases the
    external node wired to that port even though the black-box clique cannot
    see inside, so the port must be propagated as a ground source. A
    fixpoint resolves subckts that ground a port only through a nested ``X``.
    """
    # Parse each body once up front; the fixpoint below only re-runs the
    # cheap union step, never re-lexes.
    defs: dict[str, tuple[list[str], list[tuple[InstanceLine, list[str]]]]] = {}
    for card in cards:
        if card.kind != "subckt":
            continue
        try:
            sub = SubcktCard.from_card(card)
        except SpiceLexError:
            continue
        body_cards = [c for c in cards if c.scope == (*card.scope, sub.name)]
        body = [(inst, terminals) for _c, inst, terminals in _iter_instances(body_cards)]
        defs[sub.name.lower()] = ([p.lower() for p in sub.ports], body)

    sinks = _GROUND_NODES | global_nodes
    grounds: dict[str, frozenset[int]] = {name: frozenset() for name in defs}
    for _ in range(len(defs) + 1):  # bounded fixpoint over nested grounding
        changed = False
        for name, (ports, body) in defs.items():
            dsu = _DSU()
            for sink in sinks:
                dsu.union(_GROUND_SINK, sink)
            for inst, terminals in body:
                _union_instance(dsu, inst, terminals, grounds)
            groot = dsu.find(_GROUND_SINK)
            reached = frozenset(i for i, port in enumerate(ports) if dsu.find(port) == groot)
            if reached != grounds[name]:
                grounds[name] = reached
                changed = True
        if not changed:
            break
    return grounds


def _qualify(node: str, scope: tuple[str, ...], global_nodes: set[str]) -> str:
    """Key a node for the deck-wide global-reachability graph.

    Ground and global names stay shared (bare); every local name is
    namespaced by its scope so identically-named locals in different
    subckts never merge. The NUL / SOH separators cannot occur in a real
    node name, so a namespaced local can never collide with a bare global
    or the ground sentinel.
    """
    key = node.lower()
    if key in _GROUND_NODES or key in global_nodes:
        return key
    return "\x00".join(s.lower() for s in scope) + "\x01" + key


def _grounded_globals(
    cards: list[SpiceCard],
    global_nodes: set[str],
    subckt_grounds: dict[str, frozenset[int]],
) -> set[str]:
    """The ``.GLOBAL`` nodes that reach ground (node ``0``) somewhere in the
    whole deck.

    ``.GLOBAL`` is only name-scoping — it confers no DC reference — so a
    global is a valid ground sink for the bias pass *only* if it actually
    reaches node 0 through a DC-conductive element in some scope (its driver
    may live in a different scope than a given reference). A global that
    reaches ground nowhere is genuinely floating and must stay a normal node
    so the pass can flag it. Local names are scope-qualified here so they do
    not merge across subckts; globals and ground stay shared.
    """
    if not global_nodes:
        return set()
    dsu = _DSU()
    for sink in _GROUND_NODES:
        dsu.union(_GROUND_SINK, sink)
    for card, inst, terminals in _iter_instances(cards):
        qualified = [_qualify(t, card.scope, global_nodes) for t in terminals]
        _union_instance(dsu, inst, qualified, subckt_grounds)
    ground_root = dsu.find(_GROUND_SINK)
    return {g for g in global_nodes if dsu.find(g) == ground_root}


def validate_netlist_bias_topology(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag nets with no DC path to ground — undefined operating-point bias.

    Companion to the dangling-node pass (which owns degree-1 nodes): this
    finds nodes touched by two or more element terminals that still cannot
    reach ground (node ``0``) through any DC-conductive element, so their DC
    operating point is undefined.

    The graph is built by conservative over-connection (see ``_dc_edges``):
    only capacitor dielectrics, MOSFET gate oxides, and current-source
    branches are DC opens; everything else — including bias-dependent
    diodes, transistor channels, and switches — conducts. So a flagged net
    is provably floating, never a guess, which keeps the pass safe to run
    always-on. Each physically-contiguous floating domain collapses to a
    single issue. Warning severity; the caller attaches it.

    Scopes (top level and each ``.SUBCKT`` body) are handled separately,
    with the subckt's ports treated as ground-equivalent sinks (a body node
    reaching a port reaches the outside world). ``.GLOBAL`` nodes are sinks
    too, and an ``X`` instance inherits ground through any port its subckt
    body ties to node ``0`` internally. ``.asc`` connectivity is invisible
    to this text pass, so the handler runs it on ``.cir``/``.net`` only.
    """
    global_nodes = _scan_global_nodes(cards)
    subckt_grounds = _subckt_internal_grounds(cards, global_nodes)
    # .GLOBAL is name-scoping, not a ground reference: a global is a sink only
    # where it actually reaches node 0 somewhere in the deck. A floating
    # global rail stays a normal node so it can be flagged.
    grounded_globals = _grounded_globals(cards, global_nodes, subckt_grounds)
    graphs: dict[tuple[str, ...], _BiasGraph] = {}

    for card in cards:
        if card.kind != "subckt":
            continue
        try:
            sub = SubcktCard.from_card(card)
        except SpiceLexError:
            continue
        body_graph = graphs.setdefault((*card.scope, sub.name), _BiasGraph())
        body_graph.sinks.update(port.lower() for port in sub.ports)

    for card, inst, terminals in _iter_instances(cards):
        graph = graphs.setdefault(card.scope, _BiasGraph())
        prefix = inst.ref[:1].upper()
        cleaned: list[str] = []
        for idx, node in enumerate(terminals):
            if not _NON_NODE_CHARS.isdisjoint(node):
                continue
            key = node.lower()
            cleaned.append(key)
            graph.display.setdefault(key, node)
            graph.first_card.setdefault(key, card)
            graph.referrers.setdefault(key, []).append(inst.ref)
            if prefix == "M" and idx == _MOSFET_GATE_INDEX:
                graph.gate_of.setdefault(key, inst.ref)
        if cleaned:
            graph.phys_groups.append(cleaned)
        graph.sinks.update(_apply_instance_edges(graph.dsu, inst, terminals, subckt_grounds))

    issues: list[dict[str, object]] = []
    for scope, graph in graphs.items():
        for sink in graph.sinks | grounded_globals | _GROUND_NODES:
            graph.dsu.union(_GROUND_SINK, sink)
        ground_root = graph.dsu.find(_GROUND_SINK)

        # Candidates: referenced nodes with no DC path to ground and degree
        # >= 2 (degree-1 nodes are the dangling pass's job).
        candidates = {
            key
            for key, refs in graph.referrers.items()
            if len(refs) >= 2 and graph.dsu.find(key) != ground_root
        }
        if not candidates:
            continue

        # Group candidates into physical domains so an internally-AC-coupled
        # floating region reports once. Only candidate<->candidate edges
        # merge, so a domain never bridges through a grounded or degree-1
        # node — the capacitor that fragments the DC graph reunites them here.
        domain = _DSU()
        for group in graph.phys_groups:
            members = [k for k in group if k in candidates]
            for other in members[1:]:
                domain.union(members[0], other)
        islands: dict[str, list[str]] = {}
        for key in candidates:
            islands.setdefault(domain.find(key), []).append(key)

        for members in islands.values():
            issues.append(_bias_issue(scope, members, graph))
    return issues


def _bias_issue(
    scope: tuple[str, ...], members: list[str], graph: _BiasGraph
) -> dict[str, object]:
    """Build one bias-topology issue for a floating island, classifying the
    message by what the island is made of: a MOSFET gate, a current-source
    node, a purely capacitive node, or a generic multi-node domain."""
    where = f" inside .SUBCKT {scope[-1]}" if scope else ""
    rep = min(members, key=lambda k: graph.first_card[k].line_start)
    card = graph.first_card[rep]
    common: dict[str, object] = {"line": card.line_start, "directive": _card_directive(card)}

    if len(members) == 1:
        key = members[0]
        display = graph.display[key]
        refs = graph.referrers[key]
        prefixes = {r[:1].upper() for r in refs}
        if key in graph.gate_of:
            gate_ref = graph.gate_of[key]
            others = [r for r in refs if r != gate_ref]
            via = f", reached only through {', '.join(others)}" if others else ""
            return {
                **common,
                "message": (
                    f"Node '{display}' is the gate of {gate_ref} and has no DC path "
                    f"to ground{where} — the gate oxide is an open{via}. "
                    "Operating-point bias is undefined."
                ),
                "suggestion": (
                    "Add a DC bias path to the gate (e.g. a resistor to a bias node "
                    "or rail), or set its DC level explicitly."
                ),
            }
        # I, plus the current outputs of a VCCS (G) / CCCS (F): all pin
        # current, not voltage, so a node reached only through them floats.
        current_sources = prefixes & {"I", "F", "G"}
        if current_sources:
            srcs = [r for r in refs if r[:1].upper() in current_sources]
            return {
                **common,
                "message": (
                    f"Node '{display}' is driven only by current source "
                    f"{', '.join(srcs)} with no path that pins its voltage{where} — "
                    "its DC voltage is undefined."
                ),
                "suggestion": "Add a resistive or voltage path from the node to a reference node.",
            }
        if prefixes == {"C"}:
            caps = list(refs)
            return {
                **common,
                "message": (
                    f"Node '{display}' connects to ground only through capacitors "
                    f"({', '.join(caps)}){where} — no DC path, so its DC bias is undefined."
                ),
                "suggestion": "Add a DC bias path (e.g. a bias resistor to a rail or ground).",
            }
        return {
            **common,
            "message": (
                f"Node '{display}' has no DC path to ground{where} — its DC bias is undefined."
            ),
            "suggestion": "Add a DC path from the node to a reference node.",
        }

    names = ", ".join(sorted(graph.display[k] for k in members))
    return {
        **common,
        "message": (
            f"A group of {len(members)} nodes ({names}) has no DC path to ground "
            f"(node 0){where} — their DC bias is undefined."
        ),
        "suggestion": (
            "Reference the domain to ground (e.g. a high-value resistor to node 0), "
            "or confirm the isolation is intentional."
        ),
    }


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
