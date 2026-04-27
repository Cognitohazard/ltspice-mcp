"""Monte Carlo perturbation engine.

Owns the math: given component nominal values, tolerances, distributions,
and an optional seed, produce per-run perturbed values. Knows nothing
about runners, asyncio, or filesystem layout — those concerns live in
``montecarlo_runner``.

Replaces spicelib's ``Montecarlo`` because spicelib's ``_get_sim_value``
samples ``random.gauss(value, tolerance/3)`` (absolute σ instead of
multiplicative) and uses a fresh ``random.Random()`` per call (defeats
reproducibility). Owning the math gives us the intended
``value * (1 + N(0, tol/3))`` and a seeded RNG.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.lib.format import parse_spice_value

logger = logging.getLogger(__name__)


Distribution = Literal["uniform", "normal", "gaussian"]


# Component prefixes whose values can be perturbed without breaking
# semantics. Sources, controlled sources, and switches are excluded —
# their values often encode behavior, not magnitudes.
_PERTURBABLE_PREFIXES: frozenset[str] = frozenset({"R", "C", "L"})


@dataclass(frozen=True)
class ToleranceSpec:
    """Tolerance + distribution for one component."""

    tolerance: float
    distribution: Distribution


class MCSampler:
    """Per-run value perturbation with optional seed for reproducibility."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def sample(self, value: float, spec: ToleranceSpec) -> float:
        """Return a perturbed value for one nominal/tolerance pair.

        - ``normal`` / ``gaussian``: ``value * (1 + N(0, tolerance/3))``.
          ±tolerance corresponds to ±3σ, matching LTspice's ``ntol`` macro.
        - ``uniform``: ``U(value*(1-tol), value*(1+tol))``.
        """
        dist = spec.distribution.lower()
        if dist in ("normal", "gaussian"):
            return value * (1.0 + self._rng.gauss(0.0, spec.tolerance / 3.0))
        if dist == "uniform":
            return self._rng.uniform(
                value * (1.0 - spec.tolerance),
                value * (1.0 + spec.tolerance),
            )
        raise ValueError(
            f"Unknown distribution {dist!r}; expected 'uniform' or 'normal'"
        )


def expand_tolerances(
    component_refs: list[str],
    type_tolerances: dict[str, tuple[float, str]],
    component_overrides: dict[str, tuple[float, str]],
) -> dict[str, ToleranceSpec]:
    """Resolve type-prefix and per-ref rules to a flat ref -> spec map.

    Resolution order: ``component_overrides`` (specific ref) > matching
    prefix in ``type_tolerances`` > skip. Refs whose prefix isn't in
    ``_PERTURBABLE_PREFIXES`` are dropped (perturbing a voltage source
    would change the stimulus). Drops are logged at debug level so a
    user passing e.g. ``type_tolerances={"V": 0.05}`` can find them.
    """
    resolved: dict[str, ToleranceSpec] = {}
    skipped: list[str] = []
    for ref in component_refs:
        prefix = ref[0].upper() if ref else ""
        if prefix not in _PERTURBABLE_PREFIXES:
            skipped.append(ref)
            continue

        if ref in component_overrides:
            tol, dist = component_overrides[ref]
        elif prefix in type_tolerances:
            tol, dist = type_tolerances[prefix]
        else:
            continue
        resolved[ref] = ToleranceSpec(
            tolerance=tol, distribution=dist  # type: ignore[arg-type]
        )
    if skipped:
        logger.debug(
            "MC: skipped %d non-perturbable refs (prefix not in R/C/L): %s",
            len(skipped),
            ", ".join(skipped[:8]) + ("…" if len(skipped) > 8 else ""),
        )
    return resolved


def parse_value(value: str | float) -> float | None:
    """Parse a SPICE-formatted component value to float, or None.

    Wraps ``lib.format.parse_spice_value`` with two MC-specific behaviors:
    skips parametric expressions (``{Rd}``, ``table(...)``) so they
    don't get corrupted, and accepts numeric passthrough.
    """
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    if not s or "{" in s or "(" in s:
        return None
    try:
        return parse_spice_value(s)
    except ValueError:
        return None


def format_value(value: float) -> str:
    """Format a perturbed float back to a SPICE-compatible literal.

    Uses 10 significant figures so a parse → perturb → format → parse
    round-trip doesn't drift past the perturbation noise.
    """
    return f"{value:.10g}"
