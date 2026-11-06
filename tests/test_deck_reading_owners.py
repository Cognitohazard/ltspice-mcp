"""Which module answers each question about a deck's text, checked rather than documented.

A deck read line by line, a probe pattern written again, an include target
cut out of a directive by hand: each was a second answer to a question one
module owns, and each came to disagree with it. These checks keep the answers
with their owners. An allow-list entry is a decision written down with its
reason, and the lists may only shrink. A name only its owner may call, such as
the plain value reader, is banned in ``pyproject.toml`` instead.
"""

from __future__ import annotations

import ast
import functools
import re

from tests._source_scan import modules


@functools.cache
def _functions(tree: ast.AST, prefix: str = "") -> tuple[tuple[str, ast.AST], ...]:
    """Every function in ``tree`` with its qualified name.

    A tuple, because the cache hands every caller the same object and a
    generator is spent once its first caller has read it.
    """
    found: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = f"{prefix}{node.name}"
            found.append((name, node))
            found.extend(_functions(node, f"{name}."))
        elif isinstance(node, ast.ClassDef):
            found.extend(_functions(node, f"{prefix}{node.name}."))
    return tuple(found)


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
_LINE_WALKS: dict[str, str] = {}


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
        for module, tree in modules().items()
        if module != "lib/spice_lex.py"
        for name in _line_walks_in(tree)
    }
    assert walks == set(_LINE_WALKS)


def test_every_rule_sees_every_function_of_a_module():
    """The walk is cached per tree, and each rule here reads the same trees,
    so the second rule to ask must be given the functions the first was."""
    tree = modules()["lib/spice_lex.py"]
    first = [name for name, _ in _functions(tree)]
    assert "lex" in first
    assert [name for name, _ in _functions(tree)] == first


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


#: Functions that call ``spice_lex.lex`` itself, with what each lexes. ``lex``
#: reads every line of a text as a card, a deck's title and the cards after
#: its ``.END`` among them, so a whole deck is read through ``lex_deck`` or,
#: by an edit that writes every card back, ``lex_for_edit``.
_PLAIN_LEX: dict[str, str] = {
    "lib/library_parser.py:parse_library_file": "a model library, not a deck",
    "lib/lint_rules.py:_LintContext.files": (
        "the staged included files; the deck itself is lex_deck's. What LTspice "
        "reads past an included file's .END is not recorded"
    ),
    "lib/lint_rules.py:_models_from_staged_dependencies.walk": "an included file the walk reaches",
    "lib/montecarlo.py:render_variant_model_card": "one .model card",
    "lib/montecarlo.py:parse_model_params": "one .model card",
    "lib/spice_lex_ops.py:inject_card_before_end": "the one card to insert",
    "lib/spice_validator.py:validate_directive": "one directive",
    "lib/subckt_mismatch.py:_render_clone": "one subcircuit block, copied out of a file",
    "lib/subckt_mismatch.py:render": "a patched subcircuit block the plan rendered",
    "lib/variations.py:_validate_write_sets": "one card the hierarchy resolved",
    "lib/instance_targeting.py:TargetEditor._copy_includes": "one include card, rewritten",
    "lib/instance_targeting.py:TargetEditor.clone_model": "one .model card",
    "lib/instance_targeting.py:source_lineage": (
        "every card of a file, matched to its edited copy by line; which cards "
        "are read is the hierarchy's answer (lex_deck), and only its lines are looked up"
    ),
    "lib/instance_targeting.py:TargetEditor.__init__": (
        "every card of a file by line, for an edit to emit; the cards it edits are "
        "the hierarchy's (lex_deck)"
    ),
    "lib/native_inputs.py:_Sources.occurrence": (
        "every card of a staged file and its original, matched by line; only "
        "lines the hierarchy (lex_deck) resolved are looked up"
    ),
    "lib/pdk_native.py:_original_cards": (
        "every card of an original capture by line; only lines the hierarchy "
        "(lex_deck) resolved are looked up"
    ),
    "lib/deck_prep.py:_read_export": (
        "an LTspice export scrubbed for ngspice's parser: every card loses .backanno "
        "and the section sign, read or not, which changes nothing either simulator reads"
    ),
}


def _plain_lex_calls_in(tree: ast.AST) -> set[str]:
    """The functions in ``tree`` that call ``lex`` in their own body."""
    found: set[str] = set()
    for name, function in _functions(tree):
        pending = list(ast.iter_child_nodes(function))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            pending.extend(ast.iter_child_nodes(node))
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            called = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
            if called == "lex":
                found.add(name)
    return found


def test_a_deck_is_lexed_only_through_its_reading():
    callers = {
        f"{module}:{name}"
        for module, tree in modules().items()
        if module != "lib/spice_lex.py"
        for name in _plain_lex_calls_in(tree)
    }
    assert callers == set(_PLAIN_LEX)


def test_the_plain_lex_rule_catches_what_it_names():
    tree = ast.parse(
        "def deck(text):\n"
        "    return lex(text).cards\n"
        "def outer(text):\n"
        "    def inner():\n"
        "        return spice_lex.lex(text)\n"
        "    return inner\n"
        "def reading(text):\n"
        "    return lex_deck(text).cards\n"
    )
    assert _plain_lex_calls_in(tree) == {"deck", "outer.inner"}


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
        for module, tree in modules().items()
        for name, _ in _functions(tree)
        if name.rsplit(".", 1)[-1] == "_include_target"
    }
    assert cutters <= {"lib/deck_staging.py"}
    head_sets = {
        module
        for module, tree in modules().items()
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
    "lib/trace_names.py": (
        "the results side: a caller's signal name or node pair V(a,b), read against a "
        "raw's traces"
    ),
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
        for module, tree in modules().items()
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
