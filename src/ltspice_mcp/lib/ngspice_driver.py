"""The ngspice command sequence that seeds electrical input evaluation."""

from __future__ import annotations

import re

SEED_MAX = 2147483646


def validate_seed(seed: int) -> None:
    """Require an integer in 1..2147483646, the verified ngspice seed domain."""
    if type(seed) is not int or not 1 <= seed <= SEED_MAX:
        raise ValueError("ngspice seed must be an integer in 1..2147483646")


def seeded_commands(seed: int, electrical_input: str, raw_name: str) -> str:
    """Reseed immediately before source, then run and write the current plot.

    Input names use forward slashes on both platforms. Reject control-language
    quoting and line breaks rather than allowing names to introduce commands.
    Simple basenames retain the existing native driver's exact spelling.
    """
    validate_seed(seed)

    def argument(name: str) -> str:
        if not name or any(c in name for c in '\r\n\x00"\\$`'):
            raise ValueError("Unsupported ngspice driver filename")
        return name if re.fullmatch(r"[A-Za-z0-9_.:/-]+", name) else f'"{name}"'

    return (
        f"setseed {seed}\nsource {argument(electrical_input)}\n"
        f"run\nwrite {argument(raw_name)}\nquit\n"
    )
