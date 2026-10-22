"""The node LTspice's export gives each pin of a sheet.

A net with no label has no name on the sheet; LTspice's export names it
(``N001``, or ``NC_01`` for a pin on nothing). The numbering is the exporter's
own, so it is read off an export of the sheet, never predicted. Each part's
card is found by the name the exporter writes for it, and a card lists its
nodes in its symbol's SpiceOrder (recorded: ``symbol-orientation``,
``export-instance-names``, ``export-instance-prefix``).
"""

from __future__ import annotations

from collections.abc import Sequence

from ltspice_mcp.lib.netlist_graph import canon_ref
from ltspice_mcp.lib.spice_lex import iter_by_kind, lex
from ltspice_mcp.lib.spice_lex_views import read_instance


def exported_name(reference: str, letter: str) -> str:
    """The name a part's exported card carries, as ``canon_ref`` keys it.

    The exporter puts the part's element letter in front of a name that does
    not start with it (``Load`` is ``R§Load``), and a subcircuit's ``X`` in
    front of every name (``X2`` is ``X§X2`` on LTspice 26, ``XX2`` on XVII).
    """
    name = reference.casefold()
    letter = letter.casefold()
    return letter + name if letter == "x" or not name.startswith(letter) else name


def exported_pin_nodes(
    export_text: str, parts: Sequence[tuple[str, str, Sequence[str]]]
) -> dict[tuple[str, str], str]:
    """Each pin's node in ``export_text``: ``(reference, pin name) -> node``.

    ``parts`` are ``(reference, element letter, pin names in SpiceOrder)``. A
    card may hold more nodes than the part has pins (a three-pin MOSFET symbol
    is written with its source again as the bulk), and the first are the
    pins'. A part is left out when the export has no card of its name, or one
    with fewer nodes than pins (an ``R`` prefix on a four-pin block symbol
    writes two). A node keeps the spelling the card gives it: LTspice reads
    ``out`` and ``OUT`` as one node and writes each label's own.
    """
    wanted = {exported_name(reference, letter) for reference, letter, _ in parts}
    cards = {}
    # Top-level cards only: a child sheet's parts are inside .subckt bodies.
    for card in iter_by_kind(lex(export_text).cards, "instance", scope=()):
        if card.name and canon_ref(card.name) in wanted and (line := read_instance(card)):
            cards[canon_ref(card.name)] = line.nodes
    nodes: dict[tuple[str, str], str] = {}
    for reference, letter, pins in parts:
        found = cards.get(exported_name(reference, letter))
        if found is None or len(found) < len(pins):
            continue
        nodes.update(((reference, pin), node) for pin, node in zip(pins, found, strict=False))
    return nodes
