"""Which field holds a measurement's result is decided in one place.

``log_parser.measurement_answer`` decides it, and ``answered_measurements``
writes its answer onto each entry of a log's measurements; a reader takes
them through ``services.answer_measurements`` (the summary through
``build_simulation_summary``, which is handed the producer and the deck's
cards). Ruff keeps the two functions to their callers
(``[tool.ruff.lint.flake8-tidy-imports.banned-api]``). These scans hold the
rest: every scope that takes a log's decoded measurements answers them, and
nothing outside the log reader interprets how a measurement was printed.
"""

from __future__ import annotations

import ast

from tests._source_scan import modules, scopes

#: The calls that answer a log's measurements.
_ANSWERING = {"answer_measurements", "answered_measurements"}

#: Scopes that take a log's decoded measurements and pick no field of them,
#: with why.
_PICKS_NO_FIELD: dict[str, str] = {
    "tools/analyze.py:_measurement_step_plan": (
        "counts the rows of the tables to pair each with its .step line"
    ),
}


def _called(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _takes_measurements(nodes: list[ast.AST]) -> bool:
    """Whether a scope reads a decoded log section (``.value(...)`` or
    ``section_value(...)``) and names the measurements section."""
    reads = any(_called(node) in ("value", "section_value") for node in nodes)
    names = any(isinstance(node, ast.Constant) and node.value == "measurements" for node in nodes)
    return reads and names


def test_every_reader_of_a_logs_measurements_answers_them():
    unanswered = []
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            if not _takes_measurements(nodes):
                continue
            if any(_called(node) in _ANSWERING for node in nodes):
                continue
            key = f"{path}:{scope}"
            if key not in _PICKS_NO_FIELD:
                unanswered.append(key)
    assert unanswered == [], (
        "These scopes take a log's decoded measurements without naming the field "
        "holding each result: read them through services.answer_measurements "
        f"(or list a scope that picks no field in _PICKS_NO_FIELD): {unanswered}"
    )


def test_every_listed_scope_still_takes_measurements():
    """An entry whose scope no longer takes the measurements comes out."""
    taking = {
        f"{path}:{scope}"
        for path, tree in modules().items()
        for scope, nodes in scopes(tree)
        if _takes_measurements(nodes)
    }
    assert set(_PICKS_NO_FIELD) <= taking


def test_only_the_log_reader_reads_how_a_measurement_was_printed():
    """The decoded ``printed_form`` of an entry is the log reader's to read;
    every other module sees ``answer`` or ``answer_unknown`` in its place."""
    readers = {
        path
        for path, tree in modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "printed_form"
    }
    assert readers == {"lib/log_parser.py"}
