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
import re
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


# --------------------------------------------------------------------------
# Which file and section an include names is read by staging's scanner
# --------------------------------------------------------------------------


def test_an_include_target_is_read_only_by_the_staging_scanner():
    """``deck_staging.scan_include_references`` reads the file and section an
    ``.include`` or ``.lib`` names. A reader that cuts the target out of the
    directive itself takes a library's own ``.lib tt`` section declaration for
    a file to include, and reads every section of a library."""
    cutters = {
        module
        for module, tree in _modules().items()
        for name, _ in _functions(tree)
        if name.rsplit(".", 1)[-1] == "_include_target"
    }
    assert cutters <= {"lib/deck_staging.py"}
    head_sets = {
        module
        for module, tree in _modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Set)
        and ".include" in [text for element in node.elts for text in _strings(element)]
    }
    assert head_sets == {"lib/spice_lex.py"}


# --------------------------------------------------------------------------
# A probe in a card is read by the lexer's views
# --------------------------------------------------------------------------

#: A pattern for a ``V(...)`` or ``I(...)`` probe: a ``[VI]`` class, or a
#: ``v(``/``i(`` spelled out (not the end of a longer word), before an
#: escaped parenthesis.
_PROBE_PATTERN = re.compile(r"\[[VvIi]{2}\](?:\\s\*)?\\\(|(?:(?<![A-Za-z])|(?<=\\b))[VvIi]\\\(")

#: Modules that compile a probe pattern for text that is not a deck's card.
_PROBE_PATTERN_READERS: dict[str, str] = {
    "lib/services.py": "the results side: a caller's signal name, read against a raw's traces",
    "lib/decoded_raw.py": "the results side: a trace name a raw file holds",
}

_RE_FUNCTIONS = frozenset(
    {"compile", "search", "match", "fullmatch", "finditer", "findall", "sub"}
)


def _probe_patterns_in(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _RE_FUNCTIONS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.args
        and any(_PROBE_PATTERN.search(text) for text in _strings(node.args[0]))
        for node in ast.walk(tree)
    )


def test_a_probe_in_a_card_is_read_only_by_the_lexer_views():
    """``spice_lex_views.probe_refs`` reads the probes a card names, and the
    ``.save`` view the probes it saves; a second pattern elsewhere would read
    ``V(a,b)`` one way here and another there."""
    readers = {
        module
        for module, tree in _modules().items()
        if module != "lib/spice_lex_views.py" and _probe_patterns_in(tree)
    }
    assert readers == set(_PROBE_PATTERN_READERS)


def test_the_probe_pattern_rule_catches_what_it_names():
    caught = [
        r're.compile(r"\b[VI]\s*\(([^()]*)\)")',
        r're.search(r"(?i)\bv\(\w+\)", text)',
        r're.compile(r"\s*V\(\s*(\S+)\)")',
    ]
    missed = [r're.compile(r"\.save\s+(all)")', r're.compile(r"(\w+)\(")']
    assert [_probe_patterns_in(ast.parse(source)) for source in caught] == [True] * 3
    assert [_probe_patterns_in(ast.parse(source)) for source in missed] == [False] * 2
