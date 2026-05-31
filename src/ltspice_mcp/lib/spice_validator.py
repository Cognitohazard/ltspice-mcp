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

from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import ELEMENT_SPECS, InstanceLine, MeasCard

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
# ``.op`` + analysis idiom (v9-LT). ``.op`` stays in ANALYSIS_KINDS/MEAS_KINDS
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


def validate_netlist_arity(cards: list[SpiceCard]) -> list[dict[str, object]]:
    """Flag instance cards whose positional-node count is below the per-
    element minimum, and B-sources missing the ``V=``/``I=`` prefix.

    Returns dicts matching the existing ``handle_validate_netlist`` issue
    shape: ``{line, directive, message, suggestion}``. C-N4 class of
    corruption — bodies LTspice rejects at simulation time with
    ``Expected 2 node names here`` or ``Unknown parameter``.
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

        directive = card.raw_lines[0].rstrip() if card.raw_lines else card.body.strip()

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
