"""Engineering notation and formatting utilities.

Provides SPICE notation parsing for values like '1k', '10Meg', '4.7u', etc.
Used throughout the analysis tools to accept human-friendly frequency and time values.
"""

import re

# SPICE scale factors. Matching is case-insensitive to follow SPICE convention
# (LTspice, ngspice, qspice all treat suffixes as case-insensitive).
# Order matters: longer suffixes must come first so 'Meg' matches before 'm'.
# Both 'm' and 'M' mean milli per SPICE convention; mega is spelled 'Meg'.
_SCALE_FACTORS: list[tuple[str, float]] = [
    ("meg", 1e6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
]

# Numeric prefix + optional alpha tail. The tail covers both the scale suffix
# ('k', 'meg', ...) and any unit annotation that follows it ('1ms' -> tail
# 'ms', '1uF' -> tail 'uf'). Anchored so '1k1' or 'foo' don't slip through.
_NUM_TAIL_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]+)$")


def parse_spice_value(s: str) -> float:
    """Parse a SPICE notation value to float.

    Handles scale factors (case-insensitive): T, G, Meg, k, m, u, n, p, f.
    Per SPICE convention, both 'm' and 'M' mean milli (1e-3); mega is 'Meg'.
    Trailing unit annotations after a recognised suffix are ignored, so
    '1ms', '1uF', '10MegHz', '1mV' all parse — '1ms' is treated as 1e-3
    (milli + seconds annotation), not as an unknown suffix.

    Examples: '1k' -> 1000.0, '10Meg' -> 1e7, '4.7u' -> 4.7e-6,
    '1K' -> 1000.0, '1ms' -> 1e-3, '10MegHz' -> 1e7

    Args:
        s: Value string (with or without scale factor and unit annotation)

    Returns:
        Parsed float value

    Raises:
        ValueError: If string cannot be parsed as a number, or if there is
            a trailing alpha tail with no recognised SPICE suffix at its
            start (e.g. '1Hz', '1ohm' — no suffix, only a unit).
    """
    s = s.strip()

    # Try direct float conversion first
    try:
        return float(s)
    except ValueError:
        pass

    m = _NUM_TAIL_RE.match(s)
    if m is not None:
        # group(1) is always a valid float literal by construction of the regex.
        mantissa = float(m.group(1))
        tail = m.group(2).lower()
        for suffix, multiplier in _SCALE_FACTORS:
            if tail.startswith(suffix):
                return mantissa * multiplier

    raise ValueError(
        f"Cannot parse '{s}' as SPICE value. "
        f"Expected number or number with suffix: {', '.join(suf for suf, _ in _SCALE_FACTORS)}"
    )


def format_spice_value(value: float | str) -> str:
    """Render a numeric value for emission into a SPICE netlist.

    Strings pass through verbatim — callers are responsible for quoting,
    bracing, etc. Floats use ``%.10g`` so a parse → format → parse
    round-trip doesn't drift past meaningful precision.
    """
    if isinstance(value, str):
        return value
    return f"{value:.10g}"
