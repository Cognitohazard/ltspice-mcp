"""How ngspice is seeded before it evaluates a deck, and how a file name is
written into an ngspice control script: the command sequence the server's
driver runs, the seed option a deck can carry itself, and ``control_path``."""

from __future__ import annotations

import codecs
import re
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ltspice_mcp.lib.spice_lex import SpiceLexError, directive_head
from ltspice_mcp.lib.spice_lex_views import directive_args, read_instance

if TYPE_CHECKING:
    from ltspice_mcp.lib.spice_lex import EditCards, SpiceCard

SEED_MAX = 2147483646

_SEED_VALUE = re.compile(r"\bseed\s*=\s*(\S*)", re.IGNORECASE)

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


def is_option_card(card: SpiceCard) -> bool:
    """Whether ngspice takes ``card`` as an option card: a directive whose
    name starts ``.opt`` (``.option``, ``.options``, ``.opt``), in any case
    (ngspice-42 ``inp.c``)."""
    return card.kind == "directive" and directive_head(card).startswith(".opt")


def interp_option(card: SpiceCard) -> bool | None:
    """Whether the option card ``card`` sets ``interp``, which makes a batch
    run store a transient interpolated onto the print step: ngspice-42 then
    logs "Interpolated raw file data!" and declares the print step's points in
    its raw (measured: ``tests/test_time_steps_recipe.py``). False for any
    other card; None for an option card whose arguments cannot be read."""
    if not is_option_card(card):
        return False
    try:
        words = directive_args(card)
    except SpiceLexError:
        return None
    return any(word.split("=", 1)[0].strip().casefold() == "interp" for word in words)


def read_option_cards(files: Sequence[EditCards]) -> list[tuple[int, SpiceCard]]:
    """The option cards ngspice may read in a deck's files (``is_option_card``),
    each with the index of the file it is in.

    A card in a subcircuit that no ``X`` line of the files calls is left out:
    ngspice drops such a subcircuit unread, options and all, while one called
    is read (measured on ngspice-42: the seed option under kiltpsa, hsa, ltpsa
    and spice3, and interp with no mode set). Calls are matched by name across every
    file, so a subcircuit called only from an uncalled one still counts, and a
    card in a ``.lib`` section counts whichever section the deck selects
    (kiltpsa reads every section): the answer errs towards a card being read.
    """
    called = {
        view.model.casefold()
        for lexed in files
        for card in lexed.read
        if card.kind == "instance" and (card.name or "")[:1] in {"x", "X"}
        if (view := read_instance(card)) is not None and view.model
    }
    return [
        (index, card)
        for index, lexed in enumerate(files)
        for card in lexed.read
        if is_option_card(card)
        if all(name.casefold() in called for name in card.scope)
    ]


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
    if not is_option_card(card):
        return None
    match = _SEED_VALUE.search(card.body)
    return None if match is None else match.group(1)


def deck_seeds(files: Sequence[EditCards]) -> list[tuple[int, SpiceCard, str]]:
    """The seed options ngspice may read in a deck's files, as the index of
    the file each is in, its card and the value it sets (``seed_option``),
    among the option cards it reads (``read_option_cards``)."""
    return [
        (index, card, value)
        for index, card in read_option_cards(files)
        if (value := seed_option(card)) is not None
    ]


def fixed_seed(value: str) -> bool:
    """Whether a seed option's value seeds ngspice alike on every run: a
    positive decimal integer ngspice reads as an ``int``. ``random`` seeds it
    from the clock, and a quoted or braced value stops the run (measured on
    ngspice-42)."""
    return value.isdecimal() and value.isascii() and 1 <= int(value) <= 2**31 - 1
