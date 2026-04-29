"""Pre-flight validation for SPICE directives.

The simulator's measurement engine has a narrower expression vocabulary
than the waveform viewer. Directives that look syntactically correct
(e.g. ``.MEAS AC fc WHEN vdb(out)=-3``) silently fail post-hoc inside
the .log file. This module catches the common pitfalls before the
directive is written to disk.

Layer A is intentionally a narrow blocklist of patterns we've actually
hit, not an exhaustive grammar. It grows as we find more.

Each rule has:
- ``pattern``: case-insensitive regex matched against the directive text
- ``simulators``: which simulators the rule applies to (empty = all)
- ``message``: error text shown to the user
- ``suggestion``: the alternative form to use

Rules return a ``ValidationError`` with the message + suggestion when
matched. Tool layer raises ``NetlistError`` from it so the response is
structured and actionable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

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
    pattern: re.Pattern[str]
    simulators: frozenset[str]  # empty => all simulators
    message: str
    suggestion: str


# Common .MEAS pitfalls. Function names matched case-insensitively.
_MEAS_PREFIX = r"^\s*\.meas(?:ure)?\b.*?"

_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="vdb_in_meas",
        # Matches .MEAS ... vdb(...) anywhere in the directive. LTspice's
        # measurement engine accepts mag/re/im/ph but NOT vdb — that's a
        # waveform-viewer-only function.
        pattern=re.compile(_MEAS_PREFIX + r"\bvdb\s*\(", re.IGNORECASE),
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
        # `phase()` has the same waveform-viewer-only restriction.
        pattern=re.compile(_MEAS_PREFIX + r"\bphase\s*\(", re.IGNORECASE),
        simulators=frozenset({"LTspice"}),
        message=(
            "phase() is not accepted in .MEAS directives — it's a "
            "waveform-viewer-only function. Use ph() instead."
        ),
        suggestion=("Replace phase(...) with ph(...) — ph() is the .MEAS-compatible spelling."),
    ),
    _Rule(
        name="group_delay_in_meas",
        pattern=re.compile(_MEAS_PREFIX + r"\bgroup_delay\s*\(", re.IGNORECASE),
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
    Empty / whitespace-only input is a no-op.
    """
    if not directive:
        return None
    stripped = directive.lstrip()
    if not stripped:
        return None
    # All current rules target .MEAS — short-circuit other directives so
    # the common .tran/.ac/.param case skips the regex loop entirely.
    if not stripped.lower().startswith(".meas"):
        return None

    for rule in _RULES:
        if rule.simulators and simulator not in rule.simulators:
            continue
        if rule.pattern.search(directive):
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
