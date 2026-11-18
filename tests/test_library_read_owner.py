"""The one place each of two questions is answered, held by scanning the source.

What a ``.lib`` card brings in is answered by ``deck_staging.library_read``,
and only ``LibraryRead.reads`` compares a card's library section with the
section a card selects. Only ``deck_staging.scan_include_references`` splits a
``.lib`` card into a path and a section. What a power in a parameter
expression evaluates to depends on the ngspice mode, so no call of the
evaluator names a simulator as a literal outside the allow-list: the caller
passes the reader it holds, with its mode.
"""

from __future__ import annotations

import ast

from tests._source_scan import modules, scopes

#: The scopes that may compare a card's section with a selected one, with
#: the reason.
SECTION_COMPARERS = {
    "lib/deck_staging.py:LibraryRead.reads": "the one answer",
    "lib/subckt_mismatch.py:_definitions_in": (
        "finds a definition the plan chose by the section it sits in, an identity "
        "of a card, not what a .lib card brings in"
    ),
}

#: The scopes that may read the words of a ``.lib`` card by position. Only
#: the first splits a reference into a path and a section; the others read
#: a library's own section declaration (``.lib tt`` ... ``.endl tt``), whose
#: one word is a section's name.
LIB_WORD_READERS = {
    "lib/deck_staging.py:scan_include_references": "the one splitter of a reference",
    "lib/deck_staging.py:card_sections": "a section declaration's name",
    "lib/hierarchy.py:walk_active.visit": "a section declaration's name, checked for its .endl",
}

#: Calls of ``evaluate`` or ``Environment`` that may name a simulator as a
#: literal, with the reason.
LITERAL_SIMULATOR_CALLS = {
    "lib/subckt_mismatch.py:_environment": (
        "the mismatch plan serves a deck with no semantic profile (QSPICE, Xyce), "
        "whose reading of an expression is undecided; it reads as native ngspice"
    ),
}


def _strip_casefold(node: ast.AST) -> ast.AST:
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"casefold", "lower", "upper"}
    ):
        node = node.func.value
    return node


def _is_selected_section(node: ast.AST) -> bool:
    """``reference.section``, ``ref.section``, ``x.selected``: the section a
    card selects."""
    node = _strip_casefold(node)
    return isinstance(node, ast.Attribute) and (
        node.attr == "selected"
        or (
            node.attr == "section"
            and isinstance(node.value, ast.Name)
            and node.value.id in {"ref", "reference", "included", "found"}
        )
    )


def _is_owner(node: ast.AST) -> bool:
    """A card's section as ``card_sections`` names it: ``owner``,
    ``card_section``, or an item of ``sections``."""
    node = _strip_casefold(node)
    if isinstance(node, ast.Name):
        return node.id in {"owner", "card_section"}
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sections"
    )


def _section_comparisons(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(_is_owner(item) for item in operands) and any(
            _is_selected_section(item)
            or (
                isinstance(stripped := _strip_casefold(item), ast.Name)
                and stripped.id in {"section", "selected"}
            )
            for item in operands
        ):
            return True
        # ``(owner is None) != (section is None)``: a rule of its own.
        if all(isinstance(item, ast.Compare) for item in operands) and any(
            _is_owner(inner.left) for inner in operands if isinstance(inner, ast.Compare)
        ):
            return True
    return False


def test_only_one_scope_compares_a_cards_section_with_the_selected_one():
    found = {
        f"{name}:{scope}"
        for name, tree in modules().items()
        for scope, nodes in scopes(tree)
        if _section_comparisons(nodes)
    }
    assert found <= set(SECTION_COMPARERS), sorted(found - set(SECTION_COMPARERS))


def test_the_section_scan_finds_a_second_rule():
    for source in (
        "def f(owner, section):\n    return owner.casefold() != section.casefold()\n",
        "def f(owner, section):\n    return (owner is None) != (section is None)\n",
        "def f(owner, ref):\n    return owner == ref.section\n",
    ):
        (scope,) = [nodes for name, nodes in scopes(ast.parse(source)) if name == "f"]
        assert _section_comparisons(scope), source


def _reads_lib_words(nodes: list[ast.AST]) -> bool:
    """A scope that names ``.lib`` and reads a word of a card by its index
    1 or 2 (``tokens[2]``)."""
    names_lib = any(isinstance(node, ast.Constant) and node.value == ".lib" for node in nodes)
    indexes = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in (1, 2)
        and isinstance(node.value, ast.Name)
        and "token" in node.value.id
        for node in nodes
    )
    return names_lib and indexes


def test_only_the_splitter_reads_a_lib_cards_words():
    found = {
        f"{name}:{scope}"
        for name, tree in modules().items()
        for scope, nodes in scopes(tree)
        if _reads_lib_words(nodes)
    }
    assert found <= set(LIB_WORD_READERS), sorted(found - set(LIB_WORD_READERS))


def test_the_word_scan_finds_a_second_splitter():
    source = (
        "def f(tokens):\n"
        "    if tokens[0].text == '.lib':\n"
        "        return tokens[1].text, tokens[2].text\n"
    )
    (scope,) = [nodes for name, nodes in scopes(ast.parse(source)) if name == "f"]
    assert _reads_lib_words(scope)


def _literal_simulator_calls(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in {"evaluate", "Environment"}:
            continue
        for keyword in node.keywords:
            if keyword.arg == "simulator" and isinstance(keyword.value, ast.Constant):
                return True
    return False


def test_no_evaluator_call_names_a_simulator_outside_the_allow_list():
    found = {
        f"{name}:{scope}"
        for name, tree in modules().items()
        for scope, nodes in scopes(tree)
        if _literal_simulator_calls(nodes)
    }
    assert found == set(LITERAL_SIMULATOR_CALLS), sorted(found ^ set(LITERAL_SIMULATOR_CALLS))


def test_every_evaluator_call_passes_a_mode():
    """``ngbehavior`` is a required argument of both: a call without it does
    not run, so no ngspice expression is read with no mode."""
    import inspect

    from ltspice_mcp.lib.hierarchy_expr import Environment, evaluate

    for function in (evaluate, Environment.__init__):
        parameter = inspect.signature(function).parameters["ngbehavior"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
