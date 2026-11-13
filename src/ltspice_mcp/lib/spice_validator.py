"""Pre-flight validation for SPICE directives.

The simulator's measurement evaluator has a narrower expression vocabulary
than the waveform viewer. Directives that look syntactically correct
(e.g. ``.MEAS AC fc WHEN vdb(out)=-3``) silently fail post-hoc inside
the .log file. This module catches the common pitfalls before the
directive is written to disk.

Layer A is intentionally a narrow blocklist of patterns we've actually
hit, not an exhaustive grammar. It grows as we find more.

Implementation note: rules walk classified tokens (``MeasCard.function_calls``)
rather than substring-matching regex. This keeps cases like
``.MEAS WHEN x=vdb_safe`` (a variable name that happens to start with
``vdb``) from being false-flagged, and opens the interface to checks the
regex couldn't do — signal-reference resolution, analysis-kind
mismatches, etc.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from ltspice_mcp.lib.deck_nodes import GROUND_NAMES, node_kind
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    SpiceLexErrorCategory,
    directive_head,
    lex,
    shadowed_cards,
    subckt_definitions,
    tokenize_body,
    unreadable_cards,
)
from ltspice_mcp.lib.spice_lex_views import (
    ELEMENT_SPECS,
    InstanceLine,
    MeasCard,
    ModelCard,
    SubcktCard,
    body_has_stray_kv_remnant,
    global_names,
    may_connect_nodes,
    probe_refs,
    read_instance,
    unread_node_names,
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
        # LTspice's measurement evaluator accepts mag/re/im/ph but NOT vdb —
        # that's a waveform-viewer-only function.
        blocked_function="vdb",
        simulators=frozenset({"LTspice"}),
        message=(
            "vdb() is not accepted in .MEAS directives — it's a "
            "waveform-viewer-only function in LTspice. .MEAS only "
            "supports mag(), re(), im(), ph()."
        ),
        suggestion=(
            "Use mag(V(node)) and convert to dB downstream, or rely on the "
            "analyze_results 'bode_filter' recipe for −3 dB cutoffs and "
            "'bode_point' for a gain/phase reading at one frequency."
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
            "Read phase with the analyze_results 'bode_point' recipe at two "
            "closely spaced frequencies and differentiate numerically; "
            "phase comes back wrapped, so undo a ±360° step between samples."
        ),
    ),
)


#: The constants of LTspice's expression evaluator (its help lists E, pi, K and
#: Q), which neither build takes as a .meas name: LTspice 26 refuses the deck
#: ("Expected a name that is not a reserved name here.") and XVII skips the
#: measurement ('"k" is a reserved constant name.'). time, temp and boltz are
#: accepted.
MEAS_RESERVED_NAMES = frozenset({"e", "k", "pi", "q"})


def meas_name_refused(card: SpiceCard, simulator: str = "LTspice") -> ValidationError | None:
    """The error for a lexed ``.meas`` card whose name ``simulator`` refuses."""
    name = card.name or ""
    if simulator != "LTspice" or name.casefold() not in MEAS_RESERVED_NAMES:
        return None
    names = ", ".join(sorted(MEAS_RESERVED_NAMES))
    return ValidationError(
        rule_name="meas_reserved_name",
        message=(
            f'"{name}" is one of the constants LTspice\'s expression evaluator '
            f"defines ({names}), so a .meas cannot take it as a name: LTspice 26 "
            "refuses the deck and runs nothing, and LTspice XVII skips the "
            "measurement."
        ),
        suggestion=f"Rename the measurement, for example to {name}_meas.",
    )


def meas_unclosed_paren_refused(
    card: SpiceCard, simulator: str = "LTspice"
) -> ValidationError | None:
    """The error for a lexed ``.meas`` card with a parenthesis left open,
    which LTspice refuses.

    Recorded for one card (log/meas_bad_syntax, ``FIND V(out AT 0.5m``):
    LTspice 26 stopped at it before the run ('Expected ")" here.') and ran
    nothing, and XVII ran the deck without that measurement. Any ``.meas``
    with a parenthesis left open is taken the same way, since LTspice 26's
    error names the missing parenthesis, not the form around it; what XVII
    makes of another form is not recorded.
    """
    if simulator != "LTspice" or card.kind != "meas":
        return None
    try:
        tokenize_body(card.body)
    except SpiceLexError as exc:
        if exc.category is not SpiceLexErrorCategory.UNBALANCED_PAREN:
            return None
        return ValidationError(
            rule_name="meas_unclosed_paren",
            message=(
                "A parenthesis is left open in this .meas. LTspice 26 refuses such "
                "a deck before the run ('Expected \")\" here.') and runs nothing, "
                "and LTspice XVII has run one without the measurement."
            ),
            suggestion="Add the ) that closes the ( it opens.",
        )
    return None


def leading_numeric_tokens(tokens: list[str]) -> list[float]:
    """Parse leading tokens as SPICE values, stopping at the first non-numeric.

    A directive's numeric arguments precede any trailing modifier keywords
    (``uic``, ``startup``, ``dec``…) or ``.param`` expressions, which parse_value
    rejects — so a break there keeps the estimate to the resolvable prefix.
    """
    nums: list[float] = []
    for t in tokens:
        try:
            nums.append(parse_spice_value(t))
        except (ValueError, TypeError):
            break
    return nums


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


def _validate_backanno(simulator: str) -> ValidationError | None:
    """Flag ``.backanno`` when targeting ngspice.

    ``.backanno`` is an LTspice schematic back-annotation directive. ngspice has
    no such command and aborts the run with "unimplemented dot command
    '.backanno'". Only fires for the ngspice target so a clean LTspice deck
    stays clean.
    """
    if simulator != "ngspice":
        return None
    return ValidationError(
        rule_name="backanno_ngspice",
        message=(
            "ngspice does not implement .backanno and aborts the run with "
            "'unimplemented dot command .backanno'; it is an LTspice-only "
            "schematic back-annotation directive."
        ),
        suggestion=(
            "Remove the .backanno line when targeting ngspice — it has no ngspice "
            "effect. (ngspice current probing uses .options savecurrents / .probe "
            "instead.)"
        ),
    )


def step_refused(directive: str, simulator: str = "LTspice") -> ValidationError | None:
    """Refuse a ``.step`` with nothing to step: the line alone, or a parameter
    named with no values after it.

    LTspice 26 reads neither as a directive and XVII ends the run ("Empty .step
    command", "Missing step range in .STEP command"), recorded as
    ``deck/step_bare`` and ``deck/step_param_bare`` (``step-without-values``).
    What a build does with another short form is not recorded. ngspice ignores
    every ``.step`` (the ``step-ngspice`` lint rule), so this is LTspice's.
    """
    words = directive.split()
    if simulator != "LTspice" or not words or words[0].casefold() != ".step":
        return None
    if len(words) == 1:
        what = "a .step with nothing after it"
    elif len(words) == 3 and words[1].casefold() == "param":
        what = f"a .step of the parameter {words[2]} with no values"
    else:
        return None
    return ValidationError(
        rule_name="step_without_values",
        message=f"LTspice refuses {what}: both builds stop before the run.",
        suggestion=(
            "Give the values to step: '.step param r list 1k 2k' or "
            "'.step param r 1k 10k 1k', or remove the line."
        ),
    )


def validate_directive(directive: str, simulator: str = "LTspice") -> ValidationError | None:
    """Check a directive against the pre-flight rules for the given simulator.

    Returns the first matched rule's error, or None if no rule fires. The text
    is read as the cards it holds (``validate_directive_card``): a sheet's
    text may hold a ``;`` comment and line breaks, at which an export splits
    it into cards, and each card is checked as it is read. Empty or
    whitespace-only input is a no-op, and so is text the lexer cannot split:
    its faults are not a validator's to report, and a broken directive is
    not approved by a substring check either.
    """
    if not directive or not directive.strip():
        return None
    try:
        cards = lex(directive if directive.endswith("\n") else directive + "\n").cards
    except SpiceLexError:
        return None
    return next(
        (error for card in cards if (error := validate_directive_card(card, simulator))),
        None,
    )


def validate_directive_card(card: SpiceCard, simulator: str = "LTspice") -> ValidationError | None:
    """The pre-flight rules for one lexed card of a deck.

    The card is read as the simulator reads it: its ``+`` lines joined and
    its comments left out, so a ``.meas`` whose function sits on a
    continuation line is checked whole. Covers ``.MEAS`` function blocklists,
    reserved names and a parenthesis left open, the ``.tran`` zero-step
    ngspice incompatibility, the ``.backanno`` ngspice incompatibility, and a
    ``.step`` with nothing to step (``step_refused``). Any other card passes.
    """
    if card.kind == "meas":
        if (error := meas_unclosed_paren_refused(card, simulator)) is not None:
            return error
        if (error := meas_name_refused(card, simulator)) is not None:
            return error
        refused = meas_functions_refused(card, simulator)
        return refused[0][1] if refused else None
    if card.kind != "directive":
        return None
    head = directive_head(card)
    if head.startswith(".tran"):
        return _validate_tran_tstep(card.body.strip(), simulator)
    if head == ".backanno":
        return _validate_backanno(simulator)
    if head == ".step":
        return step_refused(card.body, simulator)
    return None


def meas_functions_refused(
    card: SpiceCard, simulator: str = "LTspice"
) -> list[tuple[str, ValidationError]]:
    """Each function a lexed ``.meas`` card calls that ``simulator`` does not
    take in a measurement, with its error, in rule order."""
    try:
        meas = MeasCard.from_card(card)
    except SpiceLexError:
        return []
    called = {fc.name.lower() for fc in meas.function_calls}
    return [
        (rule.blocked_function, ValidationError(rule.name, rule.message, rule.suggestion))
        for rule in _RULES
        if (not rule.simulators or simulator in rule.simulators)
        and rule.blocked_function.lower() in called
    ]


# The checks ``validate_netlist_arity`` runs, each with the severity of every
# issue it reports (the reasons sit at each check). Each check is its own lint
# rule id, with the disposition its severity names, so suppressing one never
# silences another.
ARITY_CHECKS: dict[str, Literal["error", "warning"]] = {
    "element-arity": "error",
    "bsource-value-prefix": "error",
    "value-keyword-ltspice": "error",
    "value-expression-remnant": "warning",
}


def _card_directive(card: SpiceCard) -> str:
    """Source-form display text of a card for issue payloads."""
    return card.raw_lines[0].rstrip() if card.raw_lines else card.body.strip()


def _arity_issue(check: str, card: SpiceCard, message: str, **extra: str) -> dict[str, object]:
    """One ``validate_netlist_arity`` issue, carrying its check and severity."""
    return {
        "check": check,
        "severity": ARITY_CHECKS[check],
        "line": card.line_start,
        "directive": _card_directive(card),
        "message": message,
        **extra,
    }


def validate_netlist_arity(
    cards: list[SpiceCard], *, simulator: str = "LTspice"
) -> list[dict[str, object]]:
    """Flag instance cards the simulator cannot read as written.

    Returns dicts matching the existing ``handle_validate_netlist`` issue
    shape, ``{line, directive, message, suggestion}``, plus ``check`` (a key of
    ``ARITY_CHECKS``) and the ``severity`` that check declares; a caller
    reports each issue at that severity. ``simulator`` gates the rules one
    simulator's reading decides: the C=/L= primary-value rejection is
    LTspice's, and B-source R=/P= are LTspice forms.
    """
    issues: list[dict[str, object]] = []
    # The keys a B-source's expression may sit under. LTspice adds a
    # behavioural resistor (R=) and a power sink (P=) to the voltage and
    # current forms ngspice also has.
    bsource_keys = ("V", "I", "R", "P") if simulator == "LTspice" else ("V", "I")
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

        # E/G/F/H carry two body shapes: positional gain (``E1 out 0 in 0 10``),
        # which needs spec.min_nodes positional tokens (4 for E/G, 2 for F/H),
        # and keyed behavioural (``E1 out 0 VALUE={...}``, ``POLY(...)``,
        # ``TABLE(...)``), which guarantees only the two output nodes. Every
        # other element needs its spec's nodes whatever carries its value:
        # ``InstanceLine`` reads every positional of a keyed value
        # (``V1 a b wavefile=...``) as a node.
        keyed = bool(inst.params) and spec.kv_means_params_only
        required = 2 if keyed else spec.min_nodes
        node_count = len(inst.nodes)
        # Real LTspice 26 accepts R=<value>, but rejects C=<value> and
        # L=<value> as unknown parameters. ngspice accepts all three, so this
        # rejection only applies to the LTspice target.
        if (
            prefix in ("C", "L")
            and simulator == "LTspice"
            and any(key.upper() == prefix for key in inst.params)
        ):
            rewrite = " ".join([inst.ref, *inst.nodes, inst.value or ""])
            issues.append(
                _arity_issue(
                    "value-keyword-ltspice",
                    card,
                    f"{inst.ref}: LTspice does not accept {prefix}= as "
                    f"the primary value for a {prefix}-element; use the "
                    "positional value form instead.",
                    suggestion=f"Rewrite as `{rewrite}`.",
                )
            )

        if node_count < required:
            issues.append(
                _arity_issue(
                    "element-arity",
                    card,
                    f"{inst.ref}: expected at least {required} "
                    f"positional node(s) for a {prefix}-element, got {node_count}",
                )
            )

        if prefix == "B" and not any(key.upper() in bsource_keys for key in inst.params):
            accepted = " or ".join(f"{key}=" for key in bsource_keys)
            issues.append(
                _arity_issue(
                    "bsource-value-prefix",
                    card,
                    f"{inst.ref}: B-source requires {accepted} prefix on the expression "
                    f"for {simulator} (got params {sorted(inst.params)})",
                )
            )

        # An expression with whitespace around its operators (``V = V(a) + V(b)``)
        # leaves orphan tokens after the first key=value: the lexer re-joins only
        # glued forms (``V=V(a)+V(b)``). The same shape also matches a missing-
        # operator typo (``V=V(a) V(b)``), so we cannot claim the deck runs — but
        # a valid multi-term expression does, and the one certain consequence is
        # that a schematic value-edit would rewrite only the first key=value span
        # and drop the rest. Warning, not error: it would over-block a valid
        # deck, and the edit path (set_component_value) hard-errors on it anyway.
        if body_has_stray_kv_remnant(card.body):
            issues.append(
                _arity_issue(
                    "value-expression-remnant",
                    card,
                    f"{inst.ref}: value expression is not fully parsed — orphan "
                    "tokens after the first key=value. It cannot be value-edited "
                    "via the schematic tools (only the first key=value span is "
                    "rewritten). If this is a valid multi-term expression, wrap it "
                    "in braces (e.g. V={...}); if a term was mistyped, fix the syntax.",
                )
            )

    return issues


# The names ``deck_nodes`` reads as ground, always considered connected.
_GROUND_NODES = GROUND_NAMES
# The floating- and dangling-node checks read a deck as LTspice does, whichever
# simulator the deck is for: a port named ``gnd`` or named like a ``.global``
# net is the port, where ngspice reads it as ground or the global
# (``deck_nodes``), so for an ngspice deck they can report a node floating that
# such a port ties to ground or to the global. Hierarchy discovery, which is
# told the simulator, reads it as that simulator does; tests/test_deck_nodes.py
# pins the difference.
_READ_AS = "ltspice"


def is_ground_node(name: str) -> bool:
    """Whether a deck's node name is ground, as the validator reads it."""
    return name.casefold() in _GROUND_NODES


# Characters that cannot appear in a plain node name. Positional tokens
# carrying them are value syntax that spilled into the node slots (PULSE
# fragments, braced expressions), not nodes.
_NON_NODE_CHARS = frozenset("(){}=\"'")

# How many leading positional tokens of an instance card are element
# terminals, per prefix. Kept apart from ELEMENT_SPECS, whose ``min_nodes``
# drives the arity *errors* and lists only the elements it checks: the
# dangling *warning* must know every standard prefix (a diode's, a switch's),
# since a prefix absent here would zero out its nodes' counts and fabricate
# warnings on neighbouring elements, and counts the most nodes a card has
# (a MOSFET's 4, where the arity minimum is 3).
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
    "@": 2,  # LTspice 26's frequency response analyzer, in series
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


def _instance_terminals(inst: InstanceLine) -> tuple[list[str], list[str]]:
    """Split an instance view into ``(terminal tokens, other positionals)``.

    Terminals are the nodes ``InstanceLine`` reads, up to the count this table
    gives the element's prefix: an optional node the warning does not count (a
    transistor's substrate) and every other positional (model names, area
    factors, a W switch's controlling source, POLY tails, the X subckt name,
    all tokens of an unknown prefix such as an XSPICE A-card) land in the
    second slot. Those feed the suppressor set, so a card shape this table
    mis-models can only ever hide a warning, never invent one.
    """
    prefix = inst.ref[:1].upper()

    if prefix == "X":
        # Every positional up to the subckt name is a node.
        return list(inst.nodes), [inst.model] if inst.model else []

    candidates = list(inst.nodes)
    # A W switch's controlling source is no node, and stays a word of the card.
    tail = [word for word in (inst.control, inst.model) if word is not None]
    if inst.model is None and inst.value is not None:
        # A positional value rejoins the candidates behind the nodes: the view
        # reads the second of only two positionals as the value (``V1 a b``),
        # and the slice then counts it as a node. The keyed primary-value form
        # (``R1 a b R=1k``) reads its value from the KEY=VALUE token, not a
        # positional slot, so it does not rejoin.
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


# What separates the words of a card no rule could split into parts, and the
# operators inside an expression word: a node a card names inside an
# expression (``B1 out 0 V={V(in)*2``) is one of its words too.
# A bus name keeps its brackets (``V(d[0])*2`` holds ``d[0]``), and an index
# may be a node too, so the words are taken both with and without them.
def _unread_connectors(cards: list[SpiceCard]) -> list[tuple[SpiceCard, frozenset[str]]]:
    """Each card that may connect nodes but that no rule could split into
    parts, with the names it may still connect or define
    (``spice_lex_views.unread_node_names``). The passes here may only hide a
    warning, never make one on a guess, so they count each as connected."""
    return [
        (card, unread_node_names(card))
        for card, _error in unreadable_cards(cards)
        if may_connect_nodes(card)
    ]


def _scan_global_nodes(cards: list[SpiceCard]) -> set[str]:
    """Collect lowercased node names declared via ``.GLOBAL`` directives.

    Shared by the dangling-node and bias-topology passes — a ``.GLOBAL``
    node is connected across every scope by convention, so neither pass
    should treat it as locally unterminated or floating.
    """
    nodes: set[str] = set()
    for card in cards:
        try:
            nodes.update(name.lower() for name in global_names(card))
        except SpiceLexError:
            continue
    return nodes


def _scope_ports(cards: list[SpiceCard]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Each subcircuit body's scope, and the ports the header of the
    definition in force names (``spice_lex.subckt_definitions``)."""
    ports: dict[tuple[str, ...], tuple[str, ...]] = {}
    for index in subckt_definitions(cards).values():
        card = cards[index]
        try:
            sub = SubcktCard.from_card(card)
        except SpiceLexError:
            continue
        ports[(*card.scope, sub.name)] = tuple(sub.ports)
    return ports


def _in_force(cards: list[SpiceCard]) -> list[SpiceCard]:
    """The cards a reader of bodies by scope counts: none of a subcircuit
    definition that an earlier one of its name keeps out of force
    (``spice_lex.shadowed_cards``), whose body shares that one's scope."""
    shadowed = shadowed_cards(cards)
    return [card for index, card in enumerate(cards) if index not in shadowed]


def _shared(node: str, ports: tuple[str, ...], global_nodes: set[str]) -> bool:
    """Whether ``node``, written in a scope with ``ports``, is ground or a
    ``.GLOBAL`` net, which every scope shares (``deck_nodes``)."""
    kind = node_kind(node, ports=ports, global_names=global_nodes, simulator=_READ_AS)
    return kind in ("ground", "global")


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
    cards = _in_force(cards)
    global_nodes = _scan_global_nodes(cards)
    scope_ports = _scope_ports(cards)

    # scope → lowercased node → first reference + total reference count.
    occurrences: dict[tuple[str, ...], dict[str, _NodeUse]] = {}
    # A card no rule could read may connect any word on it.
    suppressed: set[str] = {word for _card, words in _unread_connectors(cards) for word in words}

    def record(
        scope: tuple[str, ...], node: str, referrer: str, card: SpiceCard, is_port: bool = False
    ) -> None:
        if _shared(node, scope_ports.get(scope, ()), global_nodes):
            return
        key = node.lower()
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
            # Probed identifiers in an expression (a B/E/G body, a behavioural
            # parameter) are connections too: they feed the suppressor set,
            # never a terminal count.
            suppressed.update(name.lower() for probe in probe_refs(card) for name in probe.names)
        elif card.kind in ("directive", "meas"):
            # A node that exists only to be probed by a .meas/.print/.plot/.save
            # is intentional, not dangling — suppress it like instance probe refs.
            suppressed.update(name.lower() for probe in probe_refs(card) for name in probe.names)

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


# Names a directive may legitimately reference as V(...)/I(...) that are NOT
# circuit nodes: the simulator synthesizes them. ngspice/LTspice .noise outputs
# (V(onoise)/V(inoise) and their spectra/totals) and the sweep axes. Without
# these the directive-ref check false-flags a correct noise or transient deck.
_RESERVED_TRACES = frozenset(
    {
        "onoise",
        "inoise",
        "onoise_total",
        "inoise_total",
        "onoise_spectrum",
        "inoise_spectrum",
        "time",
        "frequency",
        "freq",
    }
)


def _known_names(cards: list[SpiceCard]) -> set[str]:
    """Over-approximate the set of names a directive may reference.

    Every positional token of every instance card (terminals, model names,
    values, the ref itself), every ``.SUBCKT`` port, ``.GLOBAL`` node, and
    ground — all lowercased. Deliberately a superset: a name appearing
    anywhere at element level is treated as defined, so the dangling-reference
    pass can only fire on a name that is genuinely absent, never on one it
    failed to classify.
    """
    known: set[str] = set(_GROUND_NODES) | set(_RESERVED_TRACES)
    known |= _scan_global_nodes(cards)
    # A card no rule could read may define any word on it.
    known |= {word for _card, words in _unread_connectors(cards) for word in words}
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
    reported once. The caller attaches severity. ``cards`` are the cards the
    simulator reads (``spice_lex.lex_deck``).
    """
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
        for probe in probe_refs(card):
            for name in probe.names:
                if not _PLAIN_REF_RE.fullmatch(name):
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
                            f"References {probe.kind}({name}) but no node or device named "
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


#: The grounded ports of a subckt that cannot be read whole
#: (``_subckt_internal_grounds``): any port may reach ground inside it, so an
#: ``X`` of it counts every node it is wired to as grounded.
_EVERY_PORT: frozenset[int] = frozenset({-1})


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
        if nested is _EVERY_PORT:
            return [terminal.lower() for terminal in terminals]
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
    # A subckt whose header, or a card in whose body that may connect nodes,
    # cannot be split into parts may tie any port to ground.
    unread: set[str] = set()
    for card in cards:
        if card.kind != "subckt":
            continue
        try:
            sub = SubcktCard.from_card(card)
        except SpiceLexError:
            if card.name:
                unread.add(card.name.lower())
            continue
        body_cards = [c for c in cards if c.scope == (*card.scope, sub.name)]
        if _unread_connectors(body_cards):
            unread.add(sub.name.lower())
            continue
        body = [(inst, terminals) for _c, inst, terminals in _iter_instances(body_cards)]
        defs[sub.name.lower()] = ([p.lower() for p in sub.ports], body)
    for name in unread:
        defs.pop(name, None)

    grounds: dict[str, frozenset[int]] = {name: frozenset() for name in defs}
    grounds.update(dict.fromkeys(unread, _EVERY_PORT))
    for _ in range(len(defs) + 1):  # bounded fixpoint over nested grounding
        changed = False
        for name, (ports, body) in defs.items():
            dsu = _DSU()
            # Ground and the globals, but not a port named like either.
            for sink in _GROUND_NODES | global_nodes:
                if _shared(sink, tuple(ports), global_nodes):
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


def _qualify(
    node: str, scope: tuple[str, ...], global_nodes: set[str], ports: tuple[str, ...]
) -> str:
    """Key a node, written in ``scope`` whose ports are ``ports``, for the
    deck-wide global-reachability graph.

    Ground and global names stay shared (bare); every local name, a port
    included, is namespaced by its scope so identically-named locals in
    different subckts never merge. The NUL / SOH separators cannot occur in a
    real node name, so a namespaced local can never collide with a bare
    global or the ground sentinel.
    """
    key = node.lower()
    if _shared(node, ports, global_nodes):
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
    scope_ports = _scope_ports(cards)
    for card, inst, terminals in _iter_instances(cards):
        ports = scope_ports.get(card.scope, ())
        qualified = [_qualify(t, card.scope, global_nodes, ports) for t in terminals]
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
    cards = _in_force(cards)
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

    # A card no rule could read may connect any word on it, to ground among
    # them, so in keeping with the over-connection each of its words is a
    # sink: a net it touches is never flagged on a guess. A .SUBCKT header's
    # words are its body's ports, and a .GLOBAL's are sinks in every scope.
    unread_globals: set[str] = set()
    for card, words in _unread_connectors(cards):
        if card.kind == "instance":
            graphs.setdefault(card.scope, _BiasGraph()).sinks.update(words)
        elif card.kind == "subckt":
            if card.name:
                graphs.setdefault((*card.scope, card.name), _BiasGraph()).sinks.update(words)
        else:
            unread_globals |= words

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
        for sink in graph.sinks | grounded_globals | unread_globals | _GROUND_NODES:
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


def validate_netlist_duplicate_params(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag instance, ``.MODEL`` and ``.SUBCKT`` cards that give one parameter
    more than once.

    SPICE reads a parameter name in any case, so ``w=1u W=2u`` gives ``w``
    twice. Which copy a simulator reads is not known here, so each issue states
    the copies and nothing about their effect; the caller attaches severity.
    A card that cannot be read is reported elsewhere, and a card with fewer
    than two ``=`` cannot give a name twice.
    """
    issues: list[dict[str, object]] = []
    for card in cards:
        if card.body.count("=") < 2:
            continue
        repeated: dict[str, list[str]]
        if card.kind == "instance":
            view = read_instance(card)
            if view is None:
                continue
            subject, repeated = view.ref, view.repeated_params()
        elif card.kind in ("model", "subckt"):
            try:
                other = (ModelCard if card.kind == "model" else SubcktCard).from_card(card)
            except SpiceLexError:
                continue
            subject, repeated = other.name, other.repeated_params()
        else:
            continue
        for key, given in repeated.items():
            issues.append(
                {
                    "line": card.line_start,
                    "directive": _card_directive(card),
                    "message": f"{subject} gives {key} {len(given)} times: {', '.join(given)}.",
                    "suggestion": "Which copy the simulator reads is not known here; keep one.",
                }
            )
    return issues
