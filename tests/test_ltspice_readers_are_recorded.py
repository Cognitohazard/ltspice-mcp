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
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ltspice_mcp"
CASES = Path(__file__).resolve().parent / "fixtures" / "ltspice_recorded" / "inputs" / "cases.toml"

#: The words that start a string literal only where code reads or writes the
#: LTspice format they belong to.
FORMAT_WORDS: dict[str, tuple[str, ...]] = {
    "sheet and symbol records (.asc, .asy)": (
        "SYMATTR",
        "PINATTR",
        "PIN ",
        "WIRE ",
        "FLAG ",
        "SYMBOL ",
        "WINDOW ",
        "IOPIN ",
        "SHEET ",
    ),
    "raw file header (.raw)": ("No. Variables", "No. Points", "Plotname", "Binary:", "Values:"),
    "log (.log)": ("Circuit:", "Semiconductor Device Operating Points:", "Fourier components of"),
}


def _words_in(path: Path) -> set[str]:
    words = {word.strip() for group in FORMAT_WORDS.values() for word in group}
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
            text = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
            for word in words:
                if (
                    text == word
                    or text.startswith((f"{word} ", f"{word}\n", f"{word}:"))
                    or (word.endswith(":") and text.startswith(word))
                ):
                    found.add(word)
    return found


def _readers() -> dict[str, set[str]]:
    return {
        path.relative_to(PACKAGE).as_posix(): words
        for path in sorted(PACKAGE.rglob("*.py"))
        if (words := _words_in(path))
    }


def _modelled() -> set[str]:
    cases = tomllib.loads(CASES.read_text(encoding="utf-8"))
    return {name for behaviour in cases["behaviour"].values() for name in behaviour["model"]}


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
