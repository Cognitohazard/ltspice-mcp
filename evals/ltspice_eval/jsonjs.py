"""JSON written the way JavaScript's ``JSON.stringify`` writes it.

Claude Code hands the model ``JSON.stringify(structuredContent)``. Python's
``json.dumps`` differs from it on numbers, so a reply measured or shown
through ``json.dumps`` is not what that model read: JavaScript has one number
type, so ``1.0`` is ``1``, and it switches to exponent form at other
magnitudes (``1e-7`` but ``0.000001``; ``1e+21`` but ``100000000000000000000``).
Strings and the rest match ``json.dumps(..., ensure_ascii=False)`` with
compact separators.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any


def dumps(value: Any) -> str:
    """``value`` serialized as ``JSON.stringify(value)`` would serialize it."""
    parts: list[str] = []
    _write(value, parts)
    return "".join(parts)


def number(value: float | int) -> str:
    """A number spelled as JavaScript's ``Number.prototype.toString`` spells it.

    Non-finite values have no JSON spelling; ``JSON.stringify`` writes them
    as ``null``.
    """
    if isinstance(value, bool):
        raise TypeError("a bool is not a number here")
    if isinstance(value, int):
        value = float(value) if abs(value) >= 2**53 else value
        if isinstance(value, int):
            return str(value)
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # The shortest digits that read back as this float: the same digits
    # JavaScript picks, though Python places the decimal point differently.
    _, digit_tuple, exponent = Decimal(repr(abs(value))).normalize().as_tuple()
    assert isinstance(exponent, int)
    digits = "".join(str(d) for d in digit_tuple)
    k = len(digits)
    n = exponent + k  # value = 0.digits * 10**n
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
        e = n - 1
        body = f"{mantissa}e{'+' if e > 0 else '-'}{abs(e)}"
    return sign + body


def _write(value: Any, parts: list[str]) -> None:
    if value is None or value is True or value is False:
        parts.append(json.dumps(value))
    elif isinstance(value, (int, float)):
        parts.append(number(value))
    elif isinstance(value, str):
        parts.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, dict):
        parts.append("{")
        for index, (key, item) in enumerate(value.items()):
            if index:
                parts.append(",")
            parts.append(json.dumps(str(key), ensure_ascii=False))
            parts.append(":")
            _write(item, parts)
        parts.append("}")
    elif isinstance(value, (list, tuple)):
        parts.append("[")
        for index, item in enumerate(value):
            if index:
                parts.append(",")
            _write(item, parts)
        parts.append("]")
    else:
        raise TypeError(f"not JSON: {type(value).__name__}")
