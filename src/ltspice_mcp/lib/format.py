"""Engineering notation and formatting utilities.

Provides SPICE notation parsing for values like '1k', '10Meg', '4.7u', etc.
Used throughout the analysis tools to accept human-friendly frequency and time values.
"""

# SPICE scale factors. Matching is case-insensitive to follow SPICE convention
# (LTspice, ngspice, qspice all treat suffixes as case-insensitive).
# Order matters: longer suffixes must come first so 'Meg' matches before 'm'.
# Both 'm' and 'M' mean milli per SPICE convention; mega is spelled 'Meg'.
SCALE_FACTORS: list[tuple[str, float]] = [
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


def parse_spice_value(s: str) -> float:
    """Parse a SPICE notation value to float.

    Handles scale factors (case-insensitive): T, G, Meg, k, m, u, n, p, f.
    Per SPICE convention, both 'm' and 'M' mean milli (1e-3); mega is 'Meg'.
    Examples: '1k' -> 1000.0, '10Meg' -> 1e7, '4.7u' -> 4.7e-6, '1K' -> 1000.0

    Args:
        s: Value string (with or without scale factor)

    Returns:
        Parsed float value

    Raises:
        ValueError: If string cannot be parsed as a number
    """
    s = s.strip()

    # Try direct float conversion first
    try:
        return float(s)
    except ValueError:
        pass

    # Case-insensitive suffix match (longest suffix first)
    s_lower = s.lower()
    for suffix, multiplier in SCALE_FACTORS:
        if s_lower.endswith(suffix):
            base = s[: -len(suffix)]
            try:
                return float(base) * multiplier
            except ValueError:
                break

    raise ValueError(
        f"Cannot parse '{s}' as SPICE value. "
        f"Expected number or number with suffix: {', '.join(suf for suf, _ in SCALE_FACTORS)}"
    )
