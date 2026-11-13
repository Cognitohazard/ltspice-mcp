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
  ``gnd``. ngspice 42 reads a port named ``gnd`` as ground, in its native,
  hsa and kiltpsa modes. Whether every ngspice version reads ``gnd`` so in
  kiltpsa mode is not established, and no other mode's reading is recorded,
  so to ngspice in any mode but native and hsa a ``gnd`` is ``unestablished``:
  hierarchy discovery refuses it (:func:`unestablished_reason`), and the
  connectivity comparison and the floating- and dangling-node checks read it
  as they read every deck before they were told its reader, as LTspice does
  (:func:`settled_node_kind`), until a recording decides.
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
#: The ngspice modes whose reading of ``gnd`` is established: native and hsa.
_GND_ESTABLISHED_MODES: frozenset[str] = frozenset({"", "hsa"})

NodeKind = Literal["ground", "global", "port", "local", "unestablished"]
#: An answer every reader can act on: :func:`settled_node_kind`'s.
SettledKind = Literal["ground", "global", "port", "local"]


def node_kind(
    name: str,
    *,
    ports: Collection[str] = (),
    global_names: Collection[str] = (),
    simulator: str,
    ngbehavior: str | None = None,
) -> NodeKind:
    """What ``name`` stands for, written in a scope whose ports are ``ports``
    (none at the top level) in a deck whose ``.global`` cards list
    ``global_names``, to ``simulator`` (``"ltspice"`` or ``"ngspice"``) and,
    for ngspice, in its compatibility mode ``ngbehavior``, which it then
    requires. Names are compared in any case, as both simulators compare them.
    ``unestablished`` where that reading is not known (the module's notes)."""
    folded = name.casefold()
    reader = simulator.casefold()
    if reader == "ngspice":
        if ngbehavior is None:
            raise ValueError("an ngspice reading of a node needs its ngbehavior")
        if folded == "gnd" and ngbehavior not in _GND_ESTABLISHED_MODES:
            return "unestablished"
    is_port = any(port.casefold() == folded for port in ports)
    if is_port and folded != GROUND and reader == "ltspice":
        return "port"
    if folded in GROUND_NAMES:
        return "ground"
    if any(other.casefold() == folded for other in global_names):
        return "global"
    if is_port:
        return "port"
    return "local"


def settled_node_kind(
    name: str,
    *,
    ports: Collection[str] = (),
    global_names: Collection[str] = (),
    simulator: str,
    ngbehavior: str | None = None,
) -> SettledKind:
    """:func:`node_kind`, with an ``unestablished`` reading answered as the
    readers that cannot refuse (the connectivity comparison, the floating-
    and dangling-node checks, the save-coverage lint) answered it before they
    were told the deck's reader: as LTspice reads the name. Kept so until a
    recording decides; ``tests/test_one_name_resolver.py`` pins it."""
    kind = node_kind(
        name, ports=ports, global_names=global_names, simulator=simulator, ngbehavior=ngbehavior
    )
    if kind == "unestablished":
        kind = node_kind(name, ports=ports, global_names=global_names, simulator="ltspice")
    assert kind != "unestablished"
    return kind


def is_ground(name: str, *, simulator: str, ngbehavior: str | None = None) -> bool:
    """Whether ``name``, written at the top level of a deck, is ground to its
    reader (:func:`settled_node_kind`)."""
    return settled_node_kind(name, simulator=simulator, ngbehavior=ngbehavior) == "ground"


def unestablished_reason(name: str, ngbehavior: str | None) -> str:
    """Why ``name`` cannot be read as a net in ngspice's ``ngbehavior`` mode,
    where :func:`node_kind` answers ``unestablished``, and what to write instead."""
    reading = "version-dependent" if ngbehavior == "kiltpsa" else "unrecorded reading of the"
    return (
        f"{reading} ground alias {name!r} in {ngbehavior} mode; "
        "use explicit node 0 or inspect a supported native/hsa deck"
    )
