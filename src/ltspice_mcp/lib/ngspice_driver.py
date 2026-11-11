"""How ngspice is seeded before it evaluates a deck: the command sequence the
server's driver runs, and the seed option a deck can carry itself."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltspice_mcp.lib.spice_lex import SpiceCard

SEED_MAX = 2147483646

_OPTION_HEAD = re.compile(r"\s*\.opt", re.IGNORECASE)
_SEED_VALUE = re.compile(r"\bseed\s*=\s*(\S*)", re.IGNORECASE)


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


def seed_option(card: SpiceCard) -> str | None:
    """What an option card sets ngspice's random seed to, as written, or None.

    ngspice takes every card starting ``.opt`` as an option card and, when it
    loads the deck and before it evaluates any parameter, reads ``seed=`` on
    each one and runs the routine the ``setseed`` command runs (ngspice-42
    ``inp.c:eval_opt`` calls ``com_sseed``). Case, spaces around ``=`` and
    other options on the card do not matter (measured on ngspice-42). Where a
    deck sets it more than once the first in the deck wins, as ngspice
    collects the cards in reverse (``options.c:inp_getopts``).
    """
    if card.kind != "directive" or not _OPTION_HEAD.match(card.body):
        return None
    match = _SEED_VALUE.search(card.body)
    return None if match is None else match.group(1)


def fixed_seed(value: str) -> bool:
    """Whether a seed option's value seeds ngspice alike on every run: a
    positive decimal integer ngspice reads as an ``int``. ``random`` seeds it
    from the clock, and a quoted or braced value stops the run (measured on
    ngspice-42)."""
    return value.isdecimal() and value.isascii() and 1 <= int(value) <= 2**31 - 1
