"""A module that reads a stored deck and turns its values into numbers asks
``stored_values`` which recorded builds read those values otherwise.

The question "which recorded builds read this stored value as another number
than the server does" has one answer, ``lib/stored_values.py``, and a reply
that gives a number or a comparison read from a stored deck states it. This
scan finds every module that both decodes a deck's bytes and calls a reader
that turns a value into a number (or compares two): such a module imports
``stored_values``, or is listed below with the reason it sends no number read
from stored bytes another build reads otherwise. A new reader of a stored
deck's values therefore cannot skip the question unnoticed.
"""

from __future__ import annotations

import ast

from tests._source_scan import modules

#: The calls that decode a stored deck's bytes.
DECODERS = frozenset(
    {
        "read_spice_text",
        "read_spice_text_with_encoding",
        "decode_spice_bytes",
        "decode_spice_bytes_with_encoding",
        "decode_spice_bytes_strictly",
        "parse_hierarchy_cards",
    }
)

#: The calls that turn a value's text into a number, compare two values, or
#: read a whole deck's values (each answers through one of the former).
NUMBER_READERS = frozenset(
    {
        "parse_spice_value",
        "plain_number",
        "parse_plain_value",
        "parse_value",
        "values_equal",
        "evaluate",
        "Environment",
        "fact",
        "step_number",
        "leading_numeric_tokens",
        "numeric_params",
        "parse_model_params",
        "load_hierarchy",
        "resolve_hierarchy",
        "parse_netlist_graph",
        "read_deck",
    }
)

#: Modules that decode a deck and read numbers but need not ask, with why.
EXEMPT: dict[str, str] = {
    "lib/experiment_runner.py": (
        "decodes a staged deck for its save list; its netlist graph is the staged copy, "
        "whose micro signs staging spelled u, and staging states the source's readings"
    ),
    "lib/instance_targeting.py": (
        "resolves the hierarchy of a staged copy for a run's edits; staging states the "
        "source's readings on the run's receipt"
    ),
    "lib/metrics.py": (
        "decodes a run's staged deck only for a noise source's prefix; its numbers are "
        "the caller's arguments"
    ),
    "lib/native_inputs.py": (
        "reads the staged copies of a run's inputs, whose micro signs staging spelled u; "
        "staging states the source's readings on the run's receipt"
    ),
    "lib/result_observations.py": (
        "reads a run's staged deck, whose micro signs staging spelled u, so every build "
        "reads its numbers alike"
    ),
    "lib/schematic_ops.py": (
        "reads a sheet, not a deck, and parse_spice_value only accepts or refuses a "
        "value; no number reaches a reply"
    ),
}


def _calls(tree: ast.AST) -> set[str]:
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    return called


def _asks(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("stored_values"):
            return True
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "stored_values" for alias in node.names
        ):
            return True
    return False


def _readers() -> dict[str, bool]:
    """Every module that decodes a deck and reads numbers, and whether it asks."""
    return {
        path: _asks(tree)
        for path, tree in modules().items()
        # The answer itself, which every other reader asks.
        if path != "lib/stored_values.py"
        and _calls(tree) & DECODERS
        and _calls(tree) & NUMBER_READERS
    }


def test_every_reader_of_a_stored_decks_values_asks_stored_values_or_says_why_not():
    unasked = sorted(path for path, asks in _readers().items() if not asks)
    assert unasked == sorted(EXEMPT), (
        "a module that decodes a deck and reads its values as numbers must ask "
        "stored_values (and state its readings in the reply) or be listed in EXEMPT "
        f"with the reason: {sorted(set(unasked) - set(EXEMPT))}; "
        f"listed but no longer such a module: {sorted(set(EXEMPT) - set(unasked))}"
    )
    assert all(EXEMPT.values())


def test_the_routes_that_read_a_stored_deck_ask_stored_values():
    """The modules behind the routes that state a reading ask it, so the scan
    above is not passed by a list that grew around them."""
    readers = _readers()
    for path in ("lib/hierarchy.py", "tools/verify.py"):
        assert readers.get(path) is True, path
    asking = {path for path, tree in modules().items() if _asks(tree)}
    assert {
        "lib/hierarchy.py",
        "lib/netlist_graph.py",
        "lib/netlist_diff.py",
        "lib/deck_staging.py",
        "tools/inspect_tools.py",
        "tools/verify.py",
    } <= asking
