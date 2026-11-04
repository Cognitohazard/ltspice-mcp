"""Every module that reads one of LTspice's file formats names a recording.

Design principle "Evidence before assumption" (``docs/DESIGN.md``): the server
models only LTspice behaviour recorded from both supported builds, and a
recording comes before the code that relies on it. ``cases.toml`` names, per
behaviour, the code that models it (``model``), and
``tests/test_ltspice_recorder.py`` holds every behaviour to a recording or a
written reason. This module closes the loop from the code's side.

The definition is mechanical: a module of ``src/ltspice_mcp`` reads an LTspice
format when a string literal in it starts with a word that only that format
spells, a keyword of a sheet's or a symbol's records (``SYMATTR``,
``PINATTR``, ``WIRE``), a field of a raw file's header (``No. Variables``,
``Plotname``, ``Binary:``), or a line of a log (``Circuit:``, ``Fourier
components of``). Each such module must be named in at least one behaviour's
``model`` list. A module that reads these files only through another module is
held to the rule by that module.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

from tests import _ltspice_recorded as rec

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ltspice_mcp"

#: The words that start a string literal only where code reads or writes the
#: LTspice format they belong to: a record keyword is followed by a space or a
#: line end, and a word ending in ":" by anything.
FORMAT_WORDS: dict[str, tuple[str, ...]] = {
    "sheet and symbol records (.asc, .asy)": (
        "SYMATTR",
        "PINATTR",
        "PIN",
        "WIRE",
        "FLAG",
        "SYMBOL",
        "WINDOW",
        "IOPIN",
        "SHEET",
    ),
    "raw file header (.raw)": ("No. Variables", "No. Points", "Plotname", "Binary:", "Values:"),
    "log (.log)": (
        "Circuit:",
        "Semiconductor Device Operating Points:",
        "Fourier components of",
        "Fatal Error:",
        "Direct Newton iteration",
    ),
}

#: Each word, and the starts of a literal that spell it.
_STARTS: dict[str, tuple[str, ...]] = {
    word: (word,) if word.endswith(":") else (f"{word} ", f"{word}\n", f"{word}:")
    for group in FORMAT_WORDS.values()
    for word in group
}
#: What a regular expression puts before the text it matches at a line start.
_ANCHOR = re.compile(r"^(?:\(\?\w+\)|\^|\\s[*+])*")


def _words_in(text: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
            literal = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
            literal = _ANCHOR.sub("", literal)
            found.update(
                word
                for word, starts in _STARTS.items()
                if literal == word or literal.startswith(starts)
            )
    return found


@functools.cache
def _readers() -> dict[str, frozenset[str]]:
    readers: dict[str, frozenset[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # Parsing is the cost; most modules spell none of the words at all.
        if any(word in text for word in _STARTS) and (words := _words_in(text)):
            readers[path.relative_to(PACKAGE).as_posix()] = frozenset(words)
    return readers


def _modelled() -> set[str]:
    return {name for behaviour in rec.CASES.behaviours.values() for name in behaviour.model}


def test_the_definition_finds_the_readers_of_each_format():
    # A definition that found nothing would pass the check below.
    readers = _readers()
    for module in (
        "lib/asc_document.py",
        "lib/symbol_file.py",
        "lib/raw_header.py",
        "lib/raw_parser.py",
        "lib/log_decode.py",
    ):
        assert module in readers, sorted(readers)


def test_every_reader_of_an_ltspice_format_names_its_recording():
    modelled = _modelled()
    unrecorded = {
        module: sorted(words) for module, words in _readers().items() if module not in modelled
    }
    assert not unrecorded, (
        "These modules read an LTspice format and no behaviour in cases.toml names them "
        f"in its 'model' list: {unrecorded}. Name each in the behaviour whose recording "
        "pins what it reads, or add one."
    )
