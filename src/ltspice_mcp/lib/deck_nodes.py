"""Which net a node name written in a deck stands for.

Every reader of a deck's nodes asks this: hierarchy discovery
(``hierarchy.resolve_hierarchy``), the connectivity comparison
(``netlist_graph``) and the floating- and dangling-node checks
(``spice_validator``). They read the answer here, so one name is one net to
all of them.

- ``0`` is ground everywhere.
- To LTspice, a name a subcircuit names as a port is that port inside its
  body, whatever else the name is, ``0`` aside. Both LTspice builds record
  this for a port named ``gnd`` (``deck/gnd_alias``) and for a port named
  like a ``.global`` net (``deck/global_port``): the body's resistor to that
  port carries the current on to the node the port is wired to.
- ``gnd``, in any case, is ground too, at the top level and inside a
  subcircuit body, except, to LTspice, inside a subcircuit that names a port
  ``gnd``. ngspice 42 reads a port named ``gnd`` as ground, in its kiltpsa
  mode too. Whether every ngspice version reads ``gnd`` so in kiltpsa mode is
  not established, so hierarchy discovery refuses to guess there.
- A name a ``.global`` card lists is one net in every scope, except, to
  LTspice, inside a subcircuit that names a port so. ngspice 42 wires a
  body's ``vdd`` to the global ``vdd`` even there, not to the node its
  ``vdd`` port is given.
- Any other name is the body's port of that name, if it has one, else a net
  of its own scope.

``gnd!`` is an ordinary node: both LTspice builds (``deck/gnd_bang``) and
ngspice 42 read ``R1 a gnd! 1k`` with ``gnd!`` as a node of its own.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

#: Ground as a deck writes it, and as the readers name it.
GROUND = "0"
#: The names read as ground outside a subcircuit that has a port of the
#: name, case-folded: by LTspice (``deck/gnd_alias``) and by ngspice 42.
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
    if is_port and folded != GROUND and simulator.casefold() == "ltspice":
        return "port"
    if folded in GROUND_NAMES:
        return "ground"
    if any(other.casefold() == folded for other in global_names):
        return "global"
    if is_port:
        return "port"
    return "local"
