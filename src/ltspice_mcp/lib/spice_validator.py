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

from ltspice_mcp.lib.spice_lex import SpiceLexError, lex
from ltspice_mcp.lib.spice_lex_views import MeasCard

# Analysis directives that LTspice accepts. Used by validate_netlist to
# match ``.meas <kind>`` against the active analysis and detect duplicates.
# Note that ``.meas noise`` is not a valid form even though ``.noise`` is —
# the meas-side set is intentionally narrower.
ANALYSIS_KINDS: frozenset[str] = frozenset({"tran", "ac", "dc", "op", "noise"})
MEAS_KINDS: frozenset[str] = frozenset({"tran", "ac", "dc", "op"})

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
            "ltspice_filter_metrics for −3 dB cutoffs and "
            "ltspice_gain_at for point queries."
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
        suggestion=(
            "Replace phase(...) with ph(...) — ph() is the .MEAS-compatible spelling."
        ),
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
            "Compute group delay via ltspice_gain_at with "
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
