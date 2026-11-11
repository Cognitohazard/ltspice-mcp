"""The ngspice command sequence that seeds electrical input evaluation, and
how a file name is written into an ngspice control script."""

from __future__ import annotations

import codecs
import re
import sys

SEED_MAX = 2147483646

_PLAIN_NAME = re.compile(r"[A-Za-z0-9_.:/-]+")
_UNWRITABLE = re.compile(r"['\"$`\\{}\x00-\x1f\x7f]")


def control_path(name: str, encoding: str = "utf-8") -> str:
    """``name`` as one word of an ngspice control command (``source``,
    ``write``), for a script written in ``encoding``; ValueError when no
    spelling of it reaches ngspice intact.

    The one answer to how the server writes a file name into a ``.control``
    script. A plain name (letters, digits, ``_ . : / -``) is written as it is,
    so recorded drivers keep their bytes. Any other is single-quoted, which
    keeps spaces and punctuation (measured with ngspice-42, for ``source`` and
    ``write``, in a driver and in a deck's own ``.control`` block). Refused:
    a single quote, which no spelling keeps; ``$`` and a backtick, which the
    control language expands (a backtick runs a shell); braces, which it
    drops; a control character (a tab, vertical tab or form feed becomes a
    space, a line break or NUL ends the line); and a leading ``~``, which it
    expands to the home folder. A double quote and a backslash are refused
    too, though they survive inside single quotes: no name the server writes
    needs them, and a double quote around a name stays part of it.

    ngspice-42 stops on a deck that is not valid UTF-8 ("UTF-8 syntax error in
    input deck"), so a non-ASCII name is written only into a UTF-8 script; on
    Windows, where how ngspice reads such a name is not measured, it is
    refused. Use forward slashes on every platform.
    """
    if not name or _UNWRITABLE.search(name) or name.startswith("~"):
        raise ValueError(
            f"ngspice's control language cannot keep the file name {name!r}; use a folder "
            "and file name without quotes, $, `, backslashes, braces, control characters "
            "or a leading ~"
        )
    if not name.isascii() and (sys.platform == "win32" or codecs.lookup(encoding).name != "utf-8"):
        raise ValueError(
            f"the file name {name!r} is not ASCII, and ngspice reads a non-ASCII name "
            f"intact only from a UTF-8 script off Windows (this script is {encoding}); "
            "use an ASCII folder and file name"
        )
    return name if _PLAIN_NAME.fullmatch(name) else f"'{name}'"


def validate_seed(seed: int) -> None:
    """Require an integer in 1..2147483646, the verified ngspice seed domain."""
    if type(seed) is not int or not 1 <= seed <= SEED_MAX:
        raise ValueError("ngspice seed must be an integer in 1..2147483646")


def seeded_commands(
    seed: int, electrical_input: str, raw_name: str, encoding: str = "utf-8"
) -> str:
    """Reseed immediately before source, then run and write the current plot.

    Input names use forward slashes on both platforms and are written by
    ``control_path`` for a driver in ``encoding``, which refuses a name the
    control language would change rather than let it introduce commands.
    """
    validate_seed(seed)
    return (
        f"setseed {seed}\nsource {control_path(electrical_input, encoding)}\n"
        f"run\nwrite {control_path(raw_name, encoding)}\nquit\n"
    )
