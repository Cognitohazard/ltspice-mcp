"""Which net a node name written in a deck stands for.

Every reader of a deck's nodes asks this: hierarchy discovery
(``hierarchy.resolve_hierarchy``), the connectivity comparison
(``netlist_graph``) and the floating- and dangling-node checks
(``spice_validator``). They read the answer here, so one name is one net to
all of them.

- ``0`` is ground everywhere.
- ``gnd``, in any case, is ground too, at the top level and inside a
  subcircuit body, except inside a subcircuit that names a port ``gnd``. To
  LTspice that ``gnd`` is the port; to ngspice it is still ground. Both
  LTspice builds record this (``deck/gnd_alias``): the body's resistor to its
  ``gnd`` port carries the current on to the node the port is wired to.
  ngspice 42 reads the same deck with the resistor to ground. In ngspice's
  kiltpsa mode which ``gnd`` it reads as ground depends on its version, which
  hierarchy discovery refuses to guess; this module answers for the other
  modes.
- A name a ``.global`` card lists is one net in every scope, also where a
  subcircuit names a port so: ngspice 42 wires a body's ``vdd`` to the
  global ``vdd``, not to the node its ``vdd`` port is given. LTspice is read
  the same way until the ``deck-node-names`` recording says otherwise.
- Any other name is the body's port of that name, if it has one, else a net
  of its own scope.

``gnd!`` is an ordinary node: ngspice 42 reads ``R1 a gnd! 1k`` with
``gnd!`` as a node of its own. LTspice is read the same way until the
``deck-node-names`` recording says otherwise.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, directive_head, tokenize_body

#: Ground as a deck writes it, and as the readers name it.
GROUND = "0"
#: The names read as ground outside a subcircuit that has a port of the
#: name, case-folded: by LTspice (``deck/gnd_alias``), and by ngspice but in
#: its kiltpsa mode (above).
GROUND_NAMES: frozenset[str] = frozenset({"0", "gnd"})

NodeKind = Literal["ground", "global", "port", "local"]


def node_kind(
    name: str,
    *,
    ports: Collection[str] = (),
    global_names: Collection[str] = (),
    simulator: str,
) -> NodeKind:
    """What ``name`` stands for, written in a scope whose ports are ``ports``
    (none at the top level) in a deck whose ``.global`` cards list
    ``global_names``, to ``simulator`` (``"ltspice"`` or ``"ngspice"``).
    Names are compared in any case, as both simulators compare them."""
    folded = name.casefold()
    is_port = any(port.casefold() == folded for port in ports)
    if folded == "gnd" and simulator.casefold() == "ltspice" and is_port:
        return "port"
    if folded in GROUND_NAMES:
        return "ground"
    if any(other.casefold() == folded for other in global_names):
        return "global"
    if is_port:
        return "port"
    return "local"


def global_names(card: SpiceCard) -> tuple[str, ...]:
    """The node names a ``.global`` card lists, as written; none for any other
    card. Raises ``SpiceLexError`` for a ``.global`` card whose parts cannot be
    read, which each caller refuses or passes over by its own rule."""
    if card.kind != "directive" or directive_head(card) != ".global":
        return ()
    tokens = tokenize_body(card.body)[1:]
    names: list[str] = []
    for token in tokens:
        if token.kind == TokenKind.COMMENT_TRAIL:
            break
        if token.kind == TokenKind.BARE:
            names.append(token.text)
    return tuple(names)
