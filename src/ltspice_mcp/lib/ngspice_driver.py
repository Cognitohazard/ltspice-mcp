"""The ngspice command sequence that seeds electrical input evaluation, and
how a file name is written into an ngspice control script."""

from __future__ import annotations

import re

SEED_MAX = 2147483646

_PLAIN_NAME = re.compile(r"[A-Za-z0-9_.:/-]+")
_UNWRITABLE = re.compile(r"['\"$`\\{}\t\r\n\x00]")


def control_path(name: str) -> str:
    """``name`` as one word of an ngspice control command (``source``,
    ``write``); ValueError when no spelling of it reaches ngspice intact.

    The one answer to how the server writes a file name into a ``.control``
    script. A plain name (letters, digits, ``_ . : / -``) is written as it is,
    so recorded drivers keep their bytes. Any other is single-quoted, which
    keeps spaces and every other character but these (measured with
    ngspice-42, for ``source`` and ``write``): a single quote; a double quote,
    which ngspice keeps as part of the name, so double-quoting a name fails;
    ``$`` and a backtick, which the control language expands; a backslash;
    braces, which it drops; a tab, which it turns into a space; and a line
    break or NUL. Use forward slashes on every platform.
    """
    if not name or _UNWRITABLE.search(name):
        raise ValueError(f"no ngspice control-script spelling keeps the name {name!r}")
    return name if _PLAIN_NAME.fullmatch(name) else f"'{name}'"


def validate_seed(seed: int) -> None:
    """Require an integer in 1..2147483646, the verified ngspice seed domain."""
    if type(seed) is not int or not 1 <= seed <= SEED_MAX:
        raise ValueError("ngspice seed must be an integer in 1..2147483646")


def seeded_commands(seed: int, electrical_input: str, raw_name: str) -> str:
    """Reseed immediately before source, then run and write the current plot.

    Input names use forward slashes on both platforms and are written by
    ``control_path``, which refuses a name the control language would change
    rather than let it introduce commands.
    """
    validate_seed(seed)
    return (
        f"setseed {seed}\nsource {control_path(electrical_input)}\n"
        f"run\nwrite {control_path(raw_name)}\nquit\n"
    )
