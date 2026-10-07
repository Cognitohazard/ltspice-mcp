"""Engineering notation and formatting utilities.

Provides SPICE notation parsing for values like '1k', '10Meg', '4.7u', etc.
Used throughout the analysis tools to accept human-friendly frequency and time values.
"""

import re
from collections.abc import Container
from typing import Any


def cap_list(payload: dict[str, Any], key: str, items: list, cap: int) -> None:
    """Attach ``items`` under ``key``, bounded, with explicit truncation.

    The structured channel's one truncation convention: when the list exceeds
    ``cap``, the first ``cap`` entries are attached and ``{key}_truncated``
    carries the TOTAL count — a capped list is a surfaced fact, never silent.
    Callers guard emptiness themselves (whether an empty list is attached or
    omitted is a per-payload contract).
    """
    if len(items) > cap:
        payload[key] = items[:cap]
        payload[f"{key}_truncated"] = len(items)
    else:
        payload[key] = items


#: SI prefixes for engineering-notation display, largest first, as
#: ``(scale, prefix)``. Display only: SPICE input spells mega ``Meg`` and micro
#: ``u`` (see :data:`_SCALE_FACTORS`).
SI_PREFIXES: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)


def si_prefix(magnitude: float) -> tuple[float, str]:
    """The largest SI prefix whose scale does not exceed ``magnitude``, as
    ``(scale, prefix)``; the smallest prefix for anything below it."""
    return next(
        ((scale, prefix) for scale, prefix in SI_PREFIXES if magnitude >= scale * (1 - 1e-9)),
        SI_PREFIXES[-1],
    )


# SPICE scale factors. Matching is case-insensitive to follow SPICE convention
# (LTspice, ngspice, qspice all treat suffixes as case-insensitive).
# Order matters: longer suffixes must come first so 'Meg' matches before 'm'.
# Both 'm' and 'M' mean milli per SPICE convention; mega is spelled 'Meg'.
_SCALE_FACTORS: list[tuple[str, float]] = [
    ("meg", 1e6),
    ("mil", 25.4e-6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
]

# A number followed by a tail that starts with a letter. The micro sign (µ,
# U+00B5) is how LTspice's exporter spells 'u' in a netlist, and the Greek mu
# (μ, U+03BC) is what a keyboard produces; both are folded to 'u' before the
# suffix table is read.
_MANTISSA = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_NUM_TAIL_RE = re.compile(rf"^({_MANTISSA})([a-zA-Zµμ].*)$")
_DIGITS_RE = re.compile(r"\d*")
# A number with at most a scale suffix and nothing after it.
_SCALED_NUMBER_RE = re.compile(
    rf"{_MANTISSA}(?:{'|'.join(suffix for suffix, _ in _SCALE_FACTORS)})?", re.IGNORECASE
)
#: The micro sign (U+00B5) and the Greek mu (U+03BC).
MICRO_SIGNS = frozenset("µμ")
_MICRO_SIGNS = str.maketrans(dict.fromkeys(MICRO_SIGNS, "u"))


def is_scaled_number(text: str) -> bool:
    """``text`` is a SPICE number with at most a scale suffix (``8``, ``2.5k``,
    ``1meg``) and nothing after it.

    Stricter than ``parse_spice_value``, which also reads a unit after the
    suffix (``1uF``) and so would take a name such as ``2NPN`` for a number.
    """
    return _SCALED_NUMBER_RE.fullmatch(text) is not None


def fold_micro_sign(text: str) -> str:
    """``µ`` (U+00B5) and ``μ`` (U+03BC) as the ``u`` suffix, everything else
    in ``text`` untouched. LTspice's exporter writes the former; a keyboard
    produces the latter."""
    return text.translate(_MICRO_SIGNS)


def parse_spice_value(s: str) -> float:
    """Parse a SPICE value to a float, as LTspice reads it.

    A scale suffix follows the number, in any case: T, G, Meg, k, mil, m, u,
    n, p, f, with 'mil' 25.4e-6 and both 'm' and 'M' milli (mega is 'Meg').
    What follows is read the way LTspice 26 and XVII were recorded reading
    it (``tests/test_recorded_ltspice_decks.py``):

    - digits right after a suffix are the fraction it stands in for, and an
      'R' does the same with no scale: '1k5' is 1500, '4R7' is 4.7, '2M2' is
      2.2e-3;
    - any other letters end the number, and the rest is skipped: '1uF' is
      1e-6, '10MegHz' 1e7, '1MHz' a millihertz, '2Hz' 2, '9V1' 9.

    Raises:
        ValueError: If ``s`` does not start with a number, or the number is
            followed by something other than a letter (``8%``).
    """
    s = s.strip()

    try:
        return float(s)
    except ValueError:
        pass

    m = _NUM_TAIL_RE.match(s)
    if m is None:
        raise ValueError(
            f"Cannot parse '{s}' as SPICE value. Expected a number, optionally followed "
            f"by a suffix: {', '.join(suf for suf, _ in _SCALE_FACTORS)}"
        )
    # group(1) is always a valid float literal by construction of the regex.
    mantissa = m.group(1)
    tail = fold_micro_sign(m.group(2))
    folded = tail.lower()
    suffix, multiplier = next(
        ((suffix, scale) for suffix, scale in _SCALE_FACTORS if folded.startswith(suffix)),
        ("r", 1.0) if folded.startswith("r") else ("", None),
    )
    if multiplier is None:
        return float(mantissa)
    fraction = _DIGITS_RE.match(tail, len(suffix))
    digits = fraction.group() if fraction is not None else ""
    if digits and mantissa.isdigit():
        return float(f"{mantissa}.{digits}") * multiplier
    return float(mantissa) * multiplier


def unique_name(
    base: str, taken: Container[str], *, fold: bool = False, max_len: int | None = None
) -> str:
    """``base``, or ``base-2``, ``base-3``…, whichever comes first not in ``taken``.

    With ``fold`` the comparison ignores case, and ``taken`` holds casefolded
    names. ``max_len`` shortens ``base`` so a suffixed name still fits.
    """
    candidate, counter = base, 2
    while (candidate.casefold() if fold else candidate) in taken:
        suffix = f"-{counter}"
        stem = base if max_len is None else base[: max_len - len(suffix)]
        candidate, counter = stem + suffix, counter + 1
    return candidate


def format_spice_value(value: float | str) -> str:
    """Render a numeric value for emission into a SPICE netlist.

    Strings pass through verbatim — callers are responsible for quoting,
    bracing, etc. Floats use ``%.10g`` so a parse → format → parse
    round-trip doesn't drift past meaningful precision.
    """
    if isinstance(value, str):
        return value
    return f"{value:.10g}"
