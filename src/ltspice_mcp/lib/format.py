"""Engineering notation and formatting utilities.

Provides SPICE notation parsing for values like '1k', '10Meg', '4.7u', etc.
Used throughout the analysis tools to accept human-friendly frequency and time values.
"""

# SPICE scale factors - sorted by descending key length to match 'Meg' before 'm'
SCALE_FACTORS = {
    "Meg": 1e6,  # Must come before 'm' - case-sensitive
    "T": 1e12,
    "G": 1e9,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def parse_spice_value(s: str) -> float:
    """Parse a SPICE notation value to float.

    Handles scale factors: T, G, Meg, k, m, u, n, p, f
    Examples: '1k' -> 1000.0, '10Meg' -> 10000000.0, '4.7u' -> 4.7e-6

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

    # Try matching scale factors (sorted by descending length to handle 'Meg' before 'm')
    for suffix, multiplier in SCALE_FACTORS.items():
        if s.endswith(suffix):
            base = s[: -len(suffix)]
            try:
                return float(base) * multiplier
            except ValueError:
                break

    raise ValueError(
        f"Cannot parse '{s}' as SPICE value. "
        f"Expected number or number with suffix: {', '.join(SCALE_FACTORS.keys())}"
    )
