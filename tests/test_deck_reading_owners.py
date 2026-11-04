"""Which module answers each question about a deck's text, checked rather than documented.

A deck read line by line, a probe pattern written again, a value read with
the plain reader, an include target cut out of a directive by hand: each was a
second answer to a question one module owns, and each came to disagree with
it. These checks keep the answers with their owners. An allow-list entry is a
decision written down with its reason, and the lists may only shrink.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ltspice_mcp"


@functools.cache
def _modules() -> dict[str, ast.Module]:
    return {
        path.relative_to(PACKAGE).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE.rglob("*.py"))
    }


def _functions(tree: ast.AST, prefix: str = "") -> Iterator[tuple[str, ast.AST]]:
    """Every function in ``tree`` with its qualified name."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = f"{prefix}{node.name}"
            yield name, node
            yield from _functions(node, f"{name}.")
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node, f"{prefix}{node.name}.")


def _calls_method(node: ast.AST, method: str) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == method
        for call in ast.walk(node)
    )


def _strings(node: ast.AST) -> list[str]:
    """The string constants ``node`` is, or holds in a tuple."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Tuple):
        return [text for element in node.elts for text in _strings(element)]
    return []


# --------------------------------------------------------------------------
# A deck's cards are read through the lexer
# --------------------------------------------------------------------------

#: Functions that walk a text's lines and test each for a directive's dot or a
#: continuation's plus. Such a walk reads a deck's physical lines, so it reads
#: the title, the cards after ``.end`` and half of a continued card;
#: ``spice_lex.lex_deck`` is the reading of a deck.
_LINE_WALKS: dict[str, str] = {
    "lib/spice_validator.py:estimate_analysis_points": (
        "no caller outside the tests; port it to lex_deck before giving it one"
    ),
}


def _line_walks_in(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for name, function in _functions(tree):
        for loop in ast.walk(function):
            if not isinstance(loop, ast.For) or not _calls_method(loop.iter, "splitlines"):
                continue
            tested = [
                text
                for call in ast.walk(loop)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "startswith"
                and call.args
                for text in _strings(call.args[0])
            ]
            if any(text[:1] in {".", "+"} for text in tested):
                found.add(name)
    return found


def test_no_deck_is_read_line_by_line_outside_the_lexer():
    walks = {
        f"{module}:{name}"
        for module, tree in _modules().items()
        if module != "lib/spice_lex.py"
        for name in _line_walks_in(tree)
    }
    assert walks == set(_LINE_WALKS)


def test_the_line_walk_rule_catches_what_it_names():
    tree = ast.parse(
        "def cards(text):\n"
        "    for number, line in enumerate(text.splitlines(), 1):\n"
        "        if line.strip().startswith(('+', '*')):\n"
        "            pass\n"
        "def directives(text):\n"
        "    for line in text.splitlines():\n"
        "        if line.lstrip().lower().startswith('.tran'):\n"
        "            pass\n"
        "def comments(text):\n"
        "    for line in text.splitlines():\n"
        "        if line.startswith('*'):\n"
        "            pass\n"
    )
    assert _line_walks_in(tree) == {"cards", "directives"}


# --------------------------------------------------------------------------
# A value that runs is read as LTspice reads it
# --------------------------------------------------------------------------

#: Modules that import ``format.parse_plain_value``, which refuses ``4R7``,
#: ``1k5`` and ``2Hz``. A value a simulator runs is read with
#: ``parse_spice_value``, held to the recordings; the plain reader is for
#: telling a number from a name.
_PLAIN_VALUE_READERS: dict[str, str] = {
    "lib/variations.py": (
        "an assignment value the caller writes into a .param or a mismatch "
        "parameter must be a plain number or SI-suffixed value, whichever "
        "simulator then reads the deck"
    ),
}


def _importers_of(name: str) -> set[str]:
    return {
        module
        for module, tree in _modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names)
    }


def test_only_the_listed_modules_read_a_value_with_the_plain_reader():
    assert _importers_of("parse_plain_value") == set(_PLAIN_VALUE_READERS)
