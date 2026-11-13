"""Connectivity-level netlist comparison — the graph verifier.

``diff_circuit`` compares two circuits by *value/attribute/directive* text.
It cannot answer the question that matters after a re-synthesis or a rebuild:
*is this the same circuit, wired the same way?* Two netlists can share every
component value yet connect them differently, or connect them identically under
renamed internal nets. This module answers connectivity.

Pipeline:

- :func:`parse_netlist_graph` turns a netlist (path or text) into a
  :class:`NetlistGraph` — a list of top-level :class:`Component` plus the
  ``.SUBCKT`` definitions found in the file. It reuses the repository's
  foundation lexer (:mod:`ltspice_mcp.lib.spice_lex`) and the typed
  :class:`~ltspice_mcp.lib.spice_lex_views.InstanceLine` /
  :class:`~ltspice_mcp.lib.spice_lex_views.SubcktCard` views, so the fiddly
  SPICE node/model/value/param disambiguation lives in exactly one place.
- :func:`flatten_graph` expands every ``X`` subcircuit instance whose
  definition is present (nested instances too), assigning hierarchical device
  refs (``X1.M2``, matching this repo's operating-point device addressing) and
  unique names to each subcircuit's internal nets. An ``X`` whose subcircuit is
  *not* defined anywhere stays a leaf black box — its instance node list is
  compared directly — and its name is reported as unresolved rather than
  quietly passed over.

Definitions are not limited to the file itself: a compiler that externalizes a
block's implementation to a project-local ``.lib`` must still verify against a
reference whose ``.subckt`` is inline, so ``.include``/``.inc``/``.lib``
references are followed (resolved against the including file's own directory,
depth-bounded and cycle-guarded, as ``deck_staging`` reads them: the
``.lib file section`` form reads only that section of the file, and a library's
own ``.lib section`` line declares a section rather than naming a file). When
one name is defined more than once, the FIRST definition in textual order wins,
with its own body only, and the losers are recorded — measured behavior, not
assumption: ngspice 42 warns ``redefinition of .subckt X, ignored`` and keeps
whichever definition came first, so an inline block only wins when it precedes
the include (the usual LTspice layout), not because it is inline.
- :func:`compare_graphs` flattens both sides and reports what differs:
  components added / removed / retyped, value and parameter mismatches, node
  partition (connectivity) mismatches, anchor violations, and port-arity
  errors. It never returns a bare bool.

Ground normalization follows LTspice's export convention (verified against real
exports: LTspice emits ``0`` for ground). Which net a name stands for is read
as ``deck_nodes`` reads it: ``0`` and ``GND`` in any case are the same net,
``GND!`` is a node of its own, and inside a subcircuit body a name can stand
for a port or a ``.global`` net. All net and reference matching is
case-insensitive, per SPICE.

A transistor drawn with a three-pin symbol is exported with four nodes: a BJT's
substrate at ``0``, a MOSFET's bulk at its source. A three-node ``Q`` card is
read with its substrate at ``0``, because both LTspice builds run it as that
card. A three-node ``M`` card is left as written: with a level-1 model LTspice
26 refuses it and LTspice XVII runs it with the bulk at ground, so it is not the
card the export wrote, and the arity error that follows says to write the bulk.
Both are recorded (``deck/bjt_three_node``, ``deck/mos_three_node``).

Canonical labeling / limitations
--------------------------------
The structural equivalence verdict uses 1-dimensional Weisfeiler-Leman color
refinement (a.k.a. naive vertex classification) over the bipartite
component/net graph, with pin *roles* (drain vs gate vs source; symmetric
passive terminals collapsed) carried on the incidences so a swapped
drain/source is not mistaken for an equivalent wiring. 1-WL is a *necessary*
isomorphism test, not a sufficient one: a small family of highly symmetric,
regular graphs (e.g. two structurally distinct but cospectral ladder networks)
can share a color histogram without being isomorphic, so this module may report
``structurally_equivalent=True`` for such a pair. Distinguishing them needs
higher-dimensional WL or an explicit backtracking search; this module does
neither (no SAT solver, by design). In practice, anchors (named rails, ports,
outputs, measurement nets) break the symmetry that would otherwise defeat 1-WL,
which is why passing them is recommended for any non-trivial comparison.

The precise per-component diffs (retype, value, param) and the *named* node
partition mismatches are derived by matching components by reference name across
the two graphs. That matching is exact and deterministic when the two netlists
share reference names (the common case: two exports of the same schematic, or a
netlist before/after an edit). When references are wholly renamed between the
two sides, per-component diffs degrade to added/removed, but the WL structural
verdict is unaffected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from ltspice_mcp.errors import UnreadableCardError
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.deck_nodes import GROUND, settled_node_kind
from ltspice_mcp.lib.deck_staging import (
    card_sections,
    closure_depth,
    in_other_section,
    resolve_reference,
    scan_include_references,
)
from ltspice_mcp.lib.device_names import path_text
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import (
    SAME_VALUE_RTOL,
    fold_micro_sign,
    parse_spice_value,
    same_value,
)
from ltspice_mcp.lib.spice_lex import (
    LexResult,
    SpiceCard,
    SpiceLexError,
    copies_by_name,
    directive_head,
    iter_by_kind,
    lex_deck,
    subckt_children,
    unreadable_cards,
)
from ltspice_mcp.lib.spice_lex_ops import without_added_x, without_section_sign
from ltspice_mcp.lib.spice_lex_views import InstanceLine, SubcktCard, global_names

if TYPE_CHECKING:
    from ltspice_mcp.lib.hierarchy import SemanticProfile

# An include-path resolver: given the filesystem path an ``.include``/``.lib``
# resolved to (base directory joined with the target, or an absolute target),
# it returns the path to actually open, or ``None`` to deny the open. The graph
# never reads a denied path — the callback is the sandbox seam the tool layer
# wires to ``safe_path`` so an in-deck include cannot escape a sandbox.
IncludeResolver = Callable[[Path], "Path | None"]

# Ground as the graph names it: every name ``deck_nodes`` reads as ground
# (``0``, ``gnd`` in any case) folds to it, so a deck written with ``gnd`` and
# one written with ``0`` compare equal. Which names those are depends on the
# deck's reader (a ``hierarchy.SemanticProfile``), which every parse is told:
# to LTspice a subcircuit port named ``gnd`` is the port, and to ngspice it is
# ground.
_GROUND = GROUND

# Passive two-terminal devices whose terminals are electrically interchangeable.
# Their pins share one role so a resistor authored "backwards" (b a instead of
# a b) is still recognized as the same connection. Everything else keeps pin
# order: a MOSFET's D/G/S/B, a source's +/-, a diode's anode/cathode all matter.
_SYMMETRIC_TWO_TERMINAL: frozenset[str] = frozenset({"R", "C", "L"})

# How deep the walk follows files that pull in other files. Which file and
# section a directive names, whether a ``.lib`` inside a library declares a
# section, and where a target resolves are not restated here: they are
# ``deck_staging``'s (``scan_include_references``, ``card_sections``,
# ``resolve_reference``), imported rather than mirrored, because a second copy
# of those rules is how this module came to read a library's ``.lib tt``
# section declaration as an include of a file named ``tt``.
_MAX_INCLUDE_DEPTH = 3

#: The cards a scope holds as its directives: what a subcircuit's body or the
#: deck's top level says beside its components. A ``.subckt``/``.ends`` pair is a
#: definition of its own, and ``.end`` closes the deck.
DIRECTIVE_CARD_KINDS = frozenset({"model", "param", "meas", "directive", "control"})


class NetlistGraphError(ValueError):
    """A netlist could not be turned into a connectivity graph.

    Carries the offending source ``line`` (1-based, when known) and the raw
    ``card`` text, so a caller can point at the exact card. Raised for malformed
    cards, unbalanced delimiters, unclosed subcircuits, and subcircuit
    port-arity mismatches during flattening. Never raised to signal "the two
    graphs differ" — that is data, returned by :func:`compare_graphs`.
    """

    def __init__(self, message: str, *, line: int | None = None, card: str | None = None) -> None:
        self.line = line
        self.card = card
        parts = [message]
        if line is not None:
            parts.append(f"(line {line})")
        if card:
            snippet = card if len(card) <= 80 else card[:77] + "..."
            parts.append(f"in card {snippet!r}")
        super().__init__(" ".join(parts))


class PortArityMismatch(NetlistGraphError):
    """A subcircuit instance connects a different node count than it declares.

    Its own type so the comparison path can recover from exactly this fault —
    re-flattening the offending instances as black boxes — while every other
    flattening failure still aborts. Matching on the message text instead meant
    a reworded sentence would silently abort a comparison that used to succeed.
    """


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One element instance, projected out of a lexed card.

    ``ref`` has the LTspice ``§`` marker stripped but otherwise preserves source
    spelling; matching is case-insensitive (:func:`canon_ref`). ``nodes`` are
    canonicalized net names (ground folded to ``0``, lowercased; in a
    subcircuit body, a port named like ground keeps its name). ``type_letter``
    is the uppercased element prefix. ``model`` is the model / subcircuit name
    for M/Q/J/X; ``value`` is the passive value or source spec for R/C/L/V/I,
    the controlling source and gain for F/H, and for a W switch the source
    whose current sets it, then any state. ``params`` are the ``key=value``
    tokens in card order, a key the card gives twice listed twice.
    """

    ref: str
    type_letter: str
    nodes: tuple[str, ...]
    model: str | None
    value: str | None
    params: tuple[tuple[str, str], ...]
    line: int = 0


@dataclass(frozen=True)
class SubcktDef:
    """A ``.SUBCKT`` definition: its port order and its direct body components.

    ``components`` are the instances declared *directly* inside this subcircuit
    (nested ``.SUBCKT`` bodies are separate :class:`SubcktDef` entries, resolved
    by name during flattening). ``param_defaults`` are the ``PARAMS:`` defaults
    on the opener.
    """

    name: str
    ports: tuple[str, ...]
    components: tuple[Component, ...]
    param_defaults: tuple[tuple[str, str], ...] = ()
    #: The directive cards one level inside the body (``.param``, ``.model``,
    #: ``.meas`` and the rest): what the comparison reads as the subcircuit's
    #: own directives. A nested definition's are its own entry's.
    directives: tuple[SpiceCard, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class MissingInclude:
    """An ``.include``/``.lib`` target that could not contribute definitions.

    ``target`` is the file token exactly as the deck wrote it; ``reason`` says
    why it yielded nothing (not found, unreadable, past the depth bound, or no
    base directory because the netlist was parsed from bare text). ``line`` is
    the line of the directive that names it, or of the malformed ``.SUBCKT`` an
    included file holds, and ``source`` the included file that holds that line,
    None when the deck itself does.
    """

    target: str
    reason: str
    line: int | None = None
    source: str | None = None


@dataclass(frozen=True)
class DuplicateSubckt:
    """A subcircuit defined more than once across the deck and its includes.

    SPICE keeps the FIRST definition in textual order and ignores later ones
    (measured on ngspice 42, which warns "redefinition of .subckt X, ignored").
    ``used`` names the source that won, ``ignored`` the sources that lost --
    surfaced as a fact so the pick is never silent.
    """

    name: str
    used: str
    ignored: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateParam:
    """A card that gives one parameter more than once.

    The card is ``ref`` as written in ``subckt`` (``None`` at the top level),
    named once however many instances of that subcircuit there are. ``key`` is
    case-folded, as SPICE reads it; ``values`` are the copies in card order.
    Which copy a simulator uses is not modelled here, so the comparison holds an
    instance equal only where it is equal whichever copy is read.
    """

    ref: str
    subckt: str | None
    side: str
    key: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class UnresolvedSubckt:
    """An ``X`` instance whose subcircuit has no definition anywhere.

    Such an instance stays a black-box leaf. ``missing_includes`` carries that
    side's unusable include targets, which is what separates "a PDK model whose
    library was never referenced" (empty) from "the definition file is missing"
    (non-empty).
    """

    name: str
    side: str
    missing_includes: tuple[str, ...] = ()


@dataclass
class NetlistGraph:
    """Parsed, un-flattened netlist: top-level components + subcircuit defs.

    ``subckts`` merges definitions found inline with those pulled in through
    ``.include``/``.lib``, first-in-textual-order winning. ``missing_includes``
    and ``duplicate_subckts`` are the fact-level record of what could not be
    loaded and what was defined twice.
    """

    components: list[Component]
    subckts: dict[str, SubcktDef]
    #: Who reads the deck, whose reading of ground and ``.global`` nets the
    #: graph's nets follow (``deck_nodes``).
    reader: SemanticProfile
    warnings: list[str] = field(default_factory=list)
    missing_includes: tuple[MissingInclude, ...] = ()
    duplicate_subckts: tuple[DuplicateSubckt, ...] = ()
    #: The node names the deck's ``.global`` cards list, lowercased: one net
    #: in every subcircuit body.
    global_nets: frozenset[str] = frozenset()
    #: The deck's directive cards outside every subcircuit definition and
    #: before its ``.end``.
    directives: tuple[SpiceCard, ...] = ()
    #: The ``.include`` and ``.lib`` cards among ``directives`` and the
    #: definitions' whose file (or section) defines subcircuits and nothing
    #: else, so what each brings is those definitions.
    definition_includes: tuple[SpiceCard, ...] = ()


@dataclass(frozen=True)
class FlatComponent:
    """A leaf component after hierarchy flattening.

    ``ref`` is the hierarchical device path (``X1.M2``); ``nodes`` are the
    flattened net names (internal subcircuit nets carry a unique
    ``<instance-path>/<net>`` name so distinct instances never collide).
    """

    ref: str
    type_letter: str
    nodes: tuple[str, ...]
    model: str | None
    value: str | None
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FlatCall:
    """A subcircuit instance flattening expanded: its path as written
    (``X1.XI``), the key of the definition it calls, and the parameters it
    runs that definition with: each it gives (an ``X`` card's ``params:``,
    with or without the keyword), else the definition's header default
    (``_passed``)."""

    ref: str
    definition: str
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FlatNetlist:
    """A fully flattened netlist: leaf components and the set of nets.

    ``unresolved_subckts`` names the subcircuits an ``X`` instance asked for but
    that no inline or included definition supplied — each left as a black box.
    """

    components: tuple[FlatComponent, ...]
    nets: frozenset[str]
    unresolved_subckts: tuple[str, ...] = ()
    #: Every subcircuit instance expanded, in the order it was reached.
    calls: tuple[FlatCall, ...] = ()


# ---------------------------------------------------------------------------
# Comparison result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentDelta:
    """A component present on only one side."""

    ref: str
    type_letter: str
    detail: str


@dataclass(frozen=True)
class RenamedInstance:
    """A subcircuit instance matched across the ``X`` LTspice puts in front of it.

    LTspice exports a subcircuit symbol's instance with an ``X`` before its
    name whatever the name is: ``Xe`` is ``X§Xe`` from LTspice 24 on and ``XXe``
    from XVII. A netlist written by hand names it ``Xe``. Not a difference.
    """

    reference_ref: str
    candidate_ref: str


@dataclass(frozen=True)
class RetypeDiff:
    """A matched component whose element type or model/subckt name changed."""

    ref: str
    reference_type: str
    candidate_type: str


@dataclass(frozen=True)
class ValueDiff:
    """A matched component whose primary value differs (beyond tolerance)."""

    ref: str
    reference_value: str | None
    candidate_value: str | None


@dataclass(frozen=True)
class ParamDiff:
    """A matched component whose ``key=value`` parameter differs."""

    ref: str
    key: str
    reference_value: str | None
    candidate_value: str | None


@dataclass(frozen=True)
class NodePartitionDiff:
    """A connectivity mismatch between the two graphs' net partitions.

    The reference-net set and candidate-net set cannot be reconciled 1-to-1.
    Two self-describing shapes:

    - **fan-out** — one reference net is forced (by the component wiring) onto
      several candidate nets: ``reference_nets`` has one entry, ``candidate_nets``
      has more than one.
    - **short** — several reference nets are merged onto one candidate net:
      ``reference_nets`` has more than one, ``candidate_nets`` has one.

    Either way both sides of the collision are named, so the caller never has to
    reconstruct the other net. ``involved`` names the device pins (``R1.2``) that
    witness the constraint.
    """

    reference_nets: tuple[str, ...]
    candidate_nets: tuple[str, ...]
    involved: tuple[str, ...]


@dataclass(frozen=True)
class AnchorViolation:
    """A named anchor net does not occupy corresponding structural positions."""

    anchor: str
    detail: str


@dataclass(frozen=True)
class ArityError:
    """A terminal count that does not agree, with what disagrees in ``detail``.

    Two cases. A component matched across the two sides with a different
    terminal count: ``side`` is None and the arities are the two sides'. A
    subcircuit instance whose node count is not its subcircuit's port count,
    within one netlist: ``side`` names that netlist, ``reference_arity`` is the
    instance's node count and ``candidate_arity`` the subcircuit's port count.
    """

    ref: str
    reference_arity: int
    candidate_arity: int
    detail: str
    side: str | None = None


def _instance_port_detail(
    ref: str, nodes: tuple[str, ...], subckt: str, ports: tuple[str, ...]
) -> str:
    detail = (
        f"{ref} connects {len(nodes)} node(s) ({', '.join(nodes)}) but subcircuit "
        f"{subckt} declares {len(ports)} port(s) ({', '.join(ports)})."
    )
    repeated = [node for node in nodes if node.casefold() == subckt.casefold()]
    if repeated:
        detail += (
            f" Node {repeated[0]!r} is the subcircuit's own name, so the card names "
            "the subcircuit twice, as a symbol does that gives the name both as its "
            "value and as its model."
        )
    return detail


@dataclass
class GraphComparison:
    """The structured result of :func:`compare_graphs`.

    ``equivalent`` is the single-glance verdict: structurally isomorphic *and*
    no component, value, parameter, anchor, or arity difference.
    ``structurally_equivalent`` is the wiring-only verdict (the 1-WL histogram
    check, ground pinned, anchor labels ignored) — it can be True while
    ``equivalent`` is False when only values or anchor placement differ.

    Directives are not part of it: this module compares connectivity. The
    tools' equivalence mode (``tools/verify.py``, ``compare_equivalence``)
    adds the directive comparison of ``netlist_diff`` to this verdict, so a
    caller asking whether two decks are the same circuit goes through it.
    """

    equivalent: bool
    structurally_equivalent: bool
    added: list[ComponentDelta] = field(default_factory=list)
    removed: list[ComponentDelta] = field(default_factory=list)
    # Fact-level, like the subckt facts below: a pairing, not a difference.
    renamed: list[RenamedInstance] = field(default_factory=list)
    retyped: list[RetypeDiff] = field(default_factory=list)
    value_mismatches: list[ValueDiff] = field(default_factory=list)
    param_mismatches: list[ParamDiff] = field(default_factory=list)
    node_partition_mismatches: list[NodePartitionDiff] = field(default_factory=list)
    anchor_violations: list[AnchorViolation] = field(default_factory=list)
    arity_errors: list[ArityError] = field(default_factory=list)
    # Fact-level context, not differences: a black-boxed subcircuit, a
    # duplicate definition and a parameter given twice are all legitimate, so
    # none of them flips ``equivalent`` by itself.
    unresolved_subckts: list[UnresolvedSubckt] = field(default_factory=list)
    duplicate_subckts: list[DuplicateSubckt] = field(default_factory=list)
    duplicate_params: list[DuplicateParam] = field(default_factory=list)
    #: The subcircuit instances both sides expanded, paired (``paired_calls``):
    #: which definition stands for which, for the directive comparison to read.
    #: Not part of the reply.
    calls: list[tuple[FlatCall, FlatCall]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Project to a plain-dict payload (JSON-friendly)."""
        return {
            "equivalent": self.equivalent,
            "structurally_equivalent": self.structurally_equivalent,
            "added": [vars(d) for d in self.added],
            "removed": [vars(d) for d in self.removed],
            "renamed": [vars(d) for d in self.renamed],
            "retyped": [vars(d) for d in self.retyped],
            "value_mismatches": [vars(d) for d in self.value_mismatches],
            "param_mismatches": [vars(d) for d in self.param_mismatches],
            "node_partition_mismatches": [
                {
                    "reference_nets": list(d.reference_nets),
                    "candidate_nets": list(d.candidate_nets),
                    "involved": list(d.involved),
                }
                for d in self.node_partition_mismatches
            ],
            "anchor_violations": [vars(d) for d in self.anchor_violations],
            "arity_errors": [vars(d) for d in self.arity_errors],
            "unresolved_subckts": [
                {
                    "name": d.name,
                    "side": d.side,
                    "missing_includes": list(d.missing_includes),
                }
                for d in self.unresolved_subckts
            ],
            "duplicate_subckts": [
                {"name": d.name, "used": d.used, "ignored": list(d.ignored)}
                for d in self.duplicate_subckts
            ],
            "duplicate_params": [
                {
                    "ref": d.ref,
                    "subckt": d.subckt,
                    "side": d.side,
                    "key": d.key,
                    "values": list(d.values),
                }
                for d in self.duplicate_params
            ],
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def canon_net(name: str, reader: SemanticProfile, ports: Sequence[str] = ()) -> str:
    """Canonicalize a net name as a scope whose ports are ``ports`` reads it
    (none at the top level), to the deck's ``reader``: stripped and
    lowercased, with ground (``deck_nodes``) as ``0``."""
    n = name.strip().lower()
    kind = settled_node_kind(
        n, ports=ports, simulator=reader.simulator, ngbehavior=reader.ngbehavior
    )
    return _GROUND if kind == "ground" else n


def canon_ref(ref: str) -> str:
    """Canonicalize a reference for matching: drop the ``§`` marker
    (``without_section_sign``), lowercase."""
    return without_section_sign(ref).strip().lower()


def _normalize_value_text(text: str) -> str:
    """Micro-sign variants as ``u``, so ``2.2µ`` compares equal to ``2.2u`` as
    text as well as as a value."""
    return fold_micro_sign(text)


def values_equal(a: str | None, b: str | None, rtol: float) -> bool:
    """Compare two value/param strings token by token, each numeric within
    ``rtol`` else textually.

    A field can hold several tokens, as a source's ``5V AC 1`` does, so the two
    must have as many tokens, and each pair must match. A number is read as
    LTspice runs it (``format.parse_spice_value``, held to the recordings):
    ``1k`` == ``1000`` == ``1.0k``, ``4.7µ`` == ``4.7u`` == ``4.7e-6``, ``4R7``
    == ``4.7`` and ``1k5`` == ``1.5k``. A pair that is not two numbers (a model
    name, a part of a ``PULSE(...)`` spec) is compared without regard to case.
    """
    if a is None or b is None:
        return a == b
    tokens_a = _normalize_value_text(a).split()
    tokens_b = _normalize_value_text(b).split()
    return len(tokens_a) == len(tokens_b) and all(
        _tokens_equal(ta, tb, rtol) for ta, tb in zip(tokens_a, tokens_b, strict=True)
    )


def _tokens_equal(a: str, b: str, rtol: float) -> bool:
    try:
        va, vb = parse_spice_value(a), parse_spice_value(b)
    except ValueError:
        return a.lower() == b.lower()
    return same_value(va, vb, rtol)


def _pin_role(type_letter: str, index: int) -> int:
    """Structural role of a terminal: collapse symmetric passive pins, else keep
    order. Two-terminal R/C/L pins share role 0; every other device is ordered."""
    if type_letter in _SYMMETRIC_TWO_TERMINAL:
        return 0
    return index


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _load_cards(path_or_text: str | Path) -> tuple[LexResult, Path | None]:
    """The cards a simulator reads from a netlist given as a filesystem path or
    as raw text (``spice_lex.lex_deck``: not its title line, nor a card after
    ``.END``).

    A :class:`~pathlib.Path` is always read as a file. A ``str`` is treated as
    raw netlist text when it spans multiple lines or is long; otherwise, if it
    names an existing file, it is read — else it is lexed as text. The second
    element is the file read (``None`` for bare text).

    An ``.asc`` schematic is refused: lexed as SPICE, its ``SYMBOL``/``WIRE``
    lines read as element cards and the graph describes nothing. Its LTspice
    export is the netlist to compare.
    """
    if isinstance(path_or_text, Path):
        return _cards_from_netlist_path(path_or_text), path_or_text
    text = path_or_text
    if "\n" not in text and len(text) <= 400:
        try:
            candidate = Path(text)
            if candidate.is_file():
                return _cards_from_netlist_path(candidate), candidate
        except OSError:
            pass
    return lex_deck(text), None


def _cards_from_netlist_path(path: Path) -> LexResult:
    if is_asc(path):
        raise NetlistGraphError(
            f"{path.name} is an .asc schematic, not a netlist; compare its LTspice export"
        )
    return lex_deck(read_spice_text(path))


# Lex warnings that indicate the netlist structure itself is broken (as opposed
# to benign round-trip notes). Any of these means the derived graph would be
# wrong, so parsing raises rather than return a partial graph.
_STRUCTURAL_WARNING_MARKERS = ("unclosed", ".ends", "no preceding card")


def parse_netlist_graph(
    path_or_text: str | Path,
    *,
    reader: SemanticProfile,
    base_dir: Path | None = None,
    include_resolver: IncludeResolver | None = None,
) -> NetlistGraph:
    """Parse a netlist into a :class:`NetlistGraph` (components + subckt defs).

    Reuses the foundation lexer and typed views. Raises
    :class:`NetlistGraphError` — naming the line and card — for a malformed
    card (unbalanced ``{}``/``()``, a non-reference instance) or a structurally
    broken netlist (unclosed ``.SUBCKT``, unmatched ``.ENDS``). Never returns a
    partial graph on such input.

    Subcircuit definitions are collected in textual order from this file and
    from every ``.include``/``.inc``/``.lib`` it pulls in (resolved against the
    including file's own directory, depth-bounded and cycle-guarded); a
    ``.lib <file> <section>`` reads only that section of the file. When a name
    is defined more than once the FIRST definition wins and the rest are recorded
    in ``duplicate_subckts`` — matching the simulator, which ignores
    redefinitions (measured on ngspice 42). Includes that yield nothing are
    recorded in ``missing_includes``; a malformed *included* file degrades to
    such a record rather than failing the whole parse, while the deck's own cards
    must parse.

    ``base_dir`` overrides the directory includes resolve against — needed when
    the netlist arrives as bare text, which otherwise has no directory context.

    ``include_resolver`` gates every include/lib file open (see
    :data:`IncludeResolver`): a target it denies is recorded as a missing
    include and never read. Omitted, includes open unconditionally.

    An ``X`` instance still lacking a definition is kept as a top-level
    component and treated as a black box by :func:`flatten_graph`.

    ``reader`` is who reads the deck: which names are ground and which a
    ``.global`` net is its reading (``deck_nodes``).
    """
    result, source = _load_cards(path_or_text)
    cards = result.cards

    structural = [
        w
        for w in result.warnings
        if any(marker in w.lower() for marker in _STRUCTURAL_WARNING_MARKERS)
    ]
    if structural:
        raise NetlistGraphError("malformed netlist: " + "; ".join(structural))

    registry = _SubcktRegistry(reader=reader)
    directory = base_dir if base_dir is not None else source.parent if source else None
    _collect_subckts(
        cards,
        origin=source or (directory or Path()) / "<netlist>",
        base_dir=directory,
        depth=0,
        registry=registry,
        seen=set(),
        source="<inline>",
        strict=True,
        include_resolver=include_resolver,
    )

    components = [
        _component_from_card(card, reader) for card in iter_by_kind(cards, "instance", scope=())
    ]
    directives = tuple(
        card
        for card in cards
        if not card.scope and not card.trailing and card.kind in DIRECTIVE_CARD_KINDS
    )
    # Cards are told apart by identity: every card compared here is alive,
    # held by the deck or a definition, and so is every card the registry
    # marked, so no two share an id.
    held = {id(card) for card in directives} | {
        id(card) for definition in registry.subckts.values() for card in definition.directives
    }
    return NetlistGraph(
        components=components,
        subckts=registry.subckts,
        reader=reader,
        warnings=list(result.warnings),
        missing_includes=tuple(registry.missing),
        duplicate_subckts=registry.duplicate_records(),
        global_nets=frozenset(registry.global_nets),
        directives=directives,
        definition_includes=tuple(card for card in registry.definitions_only if id(card) in held),
    )


@dataclass
class _SubcktRegistry:
    """Accumulates subcircuit definitions in textual order, first-wins."""

    #: Who reads the deck (``parse_netlist_graph``).
    reader: SemanticProfile
    subckts: dict[str, SubcktDef] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    ignored: dict[str, list[str]] = field(default_factory=dict)
    missing: list[MissingInclude] = field(default_factory=list)
    global_nets: set[str] = field(default_factory=set)
    #: Each include card whose file defines subcircuits and nothing else
    #: (held, so its identity is its own), and whether each file (and
    #: section) read so far does.
    definitions_only: list[SpiceCard] = field(default_factory=list)
    files: dict[tuple[Path, str | None], bool] = field(default_factory=dict)

    def register(self, definition: SubcktDef, source: str) -> None:
        key = definition.name.lower()
        if key in self.subckts:
            self.ignored.setdefault(key, []).append(source)
            return
        self.subckts[key] = definition
        self.sources[key] = source

    def duplicate_records(self) -> tuple[DuplicateSubckt, ...]:
        return tuple(
            DuplicateSubckt(
                name=self.subckts[key].name,
                used=self.sources[key],
                ignored=tuple(losers),
            )
            for key, losers in sorted(self.ignored.items())
        )


def _subckt_def_from(
    cards: list[SpiceCard], opener_index: int, reader: SemanticProfile
) -> SubcktDef:
    """Project one ``.SUBCKT`` opener and the body it opens into a definition.

    The body is read by span (``spice_lex.subckt_children``): a second
    definition of the name has the same scope tuple, and its cards are no part
    of this one.
    """
    view = SubcktCard.from_card(cards[opener_index])
    children = [card for _, card in subckt_children(cards, opener_index)]
    body = [
        _component_from_card(card, reader, view.ports)
        for card in children
        if card.kind == "instance"
    ]
    return SubcktDef(
        name=view.name,
        ports=tuple(view.ports),
        components=tuple(body),
        param_defaults=tuple(view.param_defaults.items()),
        directives=tuple(card for card in children if card.kind in DIRECTIVE_CARD_KINDS),
    )


def _collect_subckts(
    cards: list[SpiceCard],
    *,
    origin: Path,
    base_dir: Path | None,
    depth: int,
    registry: _SubcktRegistry,
    seen: set[tuple[Path, str | None]],
    source: str,
    strict: bool,
    include_resolver: IncludeResolver | None = None,
    section: str | None = None,
) -> bool:
    """Walk ``cards`` in textual order, registering definitions and following
    includes; return whether they define subcircuits and nothing else.

    Order is what makes first-wins precedence correct: an include contributes its
    definitions at the position its directive appears, exactly as textual
    substitution would. ``cards`` are the ones a simulator reads, read from
    ``origin``; ``section`` names the library section a ``.lib <file>
    <section>`` selected, and a card in any other section is skipped. With
    ``strict`` the caller's own malformed card raises; for an included file
    it is downgraded to a :class:`MissingInclude` record so a third-party library
    cannot break the comparison.

    Outside every definition, a card that is no section marker (``.lib
    <name>``, ``.endl``) and no ``.end``, and an include whose file does more
    than define subcircuits, is something else; one that cannot be read is
    taken as something else too, since what it holds is not known.
    """
    try:
        # Every directive that cannot be split into parts is a malformed card
        # here, as an instance is, so the graph is never partial: the section
        # and include readers raise on an unreadable reference only.
        for _card, error in unreadable_cards(card for card in cards if card.kind == "directive"):
            raise error
        sections = card_sections(cards, origin, closure_depth(depth))
        references = {
            id(reference.card): reference
            for reference in scan_include_references(cards, origin, depth=closure_depth(depth))
        }
    except SpiceLexError as exc:
        if strict:
            bad = next((card for card in cards if card.body and card.body == exc.body), None)
            raise NetlistGraphError(
                f"malformed directive: {exc}", line=bad.line_start if bad else None
            ) from exc
        registry.missing.append(
            MissingInclude(target=source, reason=f"malformed directive: {exc}")
        )
        return False
    wanted = section.casefold() if section is not None else None
    # A section the file does not declare: what the line brings is not known.
    only_definitions = wanted is None or wanted in {
        name.casefold() for name in sections if name is not None
    }
    for index, (card, card_section) in enumerate(zip(cards, sections, strict=True)):
        if in_other_section(card_section, section):
            continue
        # Every directive here could be split into parts (checked above).
        registry.global_nets.update(name.lower() for name in global_names(card))
        if card.kind == "subckt":
            try:
                registry.register(_subckt_def_from(cards, index, registry.reader), source)
            except (SpiceLexError, NetlistGraphError) as exc:
                if strict:
                    raise NetlistGraphError(
                        # The refusal of a card that cannot be read already
                        # says which card and what is wrong.
                        str(exc)
                        if isinstance(exc, UnreadableCardError)
                        else f"malformed .SUBCKT: {exc}",
                        line=card.line_start,
                        card="".join(card.raw_lines).strip(),
                    ) from exc
                registry.missing.append(
                    MissingInclude(
                        target=source,
                        reason=f"malformed .SUBCKT: {exc}",
                        line=card.line_start,
                        source=source,
                    )
                )
                only_definitions = False
            continue
        reference = references.get(id(card))
        if reference is None:
            if not card.scope and not _definition_neutral(card):
                only_definitions = False
            continue
        defines_only = _follow_include(
            reference.raw_path,
            section=reference.section,
            base_dir=base_dir,
            depth=depth,
            registry=registry,
            seen=seen,
            include_resolver=include_resolver,
            line=card.line_start,
            # The deck's own directives are at depth 0; an included file names itself.
            source=None if depth == 0 else source,
        )
        if defines_only:
            registry.definitions_only.append(card)
        elif not card.scope:
            only_definitions = False
    return only_definitions


def _definition_neutral(card: SpiceCard) -> bool:
    """Whether a card outside every definition, and no include, says nothing
    about the circuit: a blank, a comment, ``.end``, or a library's section
    marker (``.lib <name>`` declaring one, ``.endl`` closing it)."""
    if card.kind in ("blank", "comment", "end", "subckt", "ends"):
        return True
    return directive_head(card) in (".lib", ".endl")


def _follow_include(
    target: str,
    *,
    section: str | None,
    base_dir: Path | None,
    depth: int,
    registry: _SubcktRegistry,
    seen: set[tuple[Path, str | None]],
    include_resolver: IncludeResolver | None = None,
    line: int,
    source: str | None,
) -> bool:
    """Load one include target's definitions (only ``section``'s, when the
    reference selects one), or record why it contributed none; return whether
    it defines subcircuits and nothing else (False for one not read).

    ``line`` is the line of the directive that names it, and ``source`` the
    included file holding that directive (None for the deck itself), which a
    record of why it contributed nothing carries.
    """

    def missing(reason: str) -> bool:
        registry.missing.append(MissingInclude(target, reason, line=line, source=source))
        return False

    if base_dir is None:
        return missing("no base directory (netlist parsed from text)")
    if depth >= _MAX_INCLUDE_DEPTH:
        return missing(f"include depth limit ({_MAX_INCLUDE_DEPTH}) reached")
    # One resolver for the whole product: staging's answer to "which file does
    # this reference name?" is the answer here too. A local join would read
    # `C:\...\standard.mos` — the spelling LTspice's own netlister writes — as a
    # relative name and hang it off base_dir, producing a path that exists
    # nowhere and a missing-include finding for a file staging just staged.
    path = resolve_reference(base_dir, target)
    if include_resolver is not None:
        # Gate the open BEFORE any read: a denied include contributes nothing and
        # its bytes are never touched — the sandbox property the tool layer
        # relies on. The resolver may also redirect to a canonical safe path.
        approved = include_resolver(path)
        if approved is None:
            return missing("path denied by include resolver")
        path = approved
    try:
        key = path.resolve()
    except OSError:
        return missing("unresolvable path")
    selected = (key, section.casefold() if section is not None else None)
    if selected in seen:
        # Already contributed; revisiting is how an include cycle stops. One
        # still being read is taken as doing more, since that is not known yet.
        return registry.files.get(selected, False)
    seen.add(selected)
    try:
        included = lex_deck(read_spice_text(path), title=False)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return missing(f"unreadable: {exc}")
    registry.files[selected] = _collect_subckts(
        included.cards,
        origin=path,
        section=section,
        base_dir=path.parent,
        depth=depth + 1,
        registry=registry,
        seen=seen,
        source=str(path),
        strict=False,
        include_resolver=include_resolver,
    )
    return registry.files[selected]


def _component_from_card(
    card: object, reader: SemanticProfile, ports: Sequence[str] = ()
) -> Component:
    """Build a :class:`Component` from an instance card via the typed view, in
    a scope whose ports are ``ports`` (none at the top level)."""
    try:
        line = InstanceLine.from_card(card)  # type: ignore[arg-type]
    except SpiceLexError as exc:
        raw = "".join(getattr(card, "raw_lines", [])).strip()
        raise NetlistGraphError(
            f"malformed instance: {exc}",
            line=getattr(card, "line_start", None),
            card=raw,
        ) from exc
    ref = without_section_sign(line.ref)
    return Component(
        ref=ref,
        type_letter=ref[:1].upper(),
        nodes=tuple(canon_net(n, reader, ports) for n in line.nodes),
        model=line.model,
        # A V or I source compares as display shows it: its value and its own
        # words without ``=`` (``AC 2``, ``load``), which ``params`` does not
        # hold, in card order. A W switch's controlling source goes with its
        # value.
        value=(
            line.display_value() or None
            if line.source is not None
            else " ".join(part for part in (line.control, line.value) if part) or None
        ),
        params=tuple(line.param_pairs()),
        line=getattr(card, "line_start", 0),
    )


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def flatten_graph(graph: NetlistGraph) -> FlatNetlist:
    """Expand every defined ``X`` subcircuit instance into leaf components.

    Nested instances are expanded recursively. Each subcircuit's internal nets
    (those that are neither ports nor ground) get a unique flattened name keyed
    on the instance path, so two instances of one subcircuit never share
    internal nets. Leaf device refs become hierarchical paths (``X1.M2``).

    An ``X`` instance whose subcircuit is undefined — inline and after every
    include has been followed — is emitted as a leaf black box (its instance
    nodes are the connection) and its name is reported in
    ``FlatNetlist.unresolved_subckts`` rather than passed over silently.

    Raises :class:`NetlistGraphError` on a port-arity mismatch (instance node
    count vs. subcircuit port count) or a recursive subcircuit definition.
    """
    return _flatten(graph, arity=None)[0]


def _flatten(
    graph: NetlistGraph, *, arity: list[ArityError] | None
) -> tuple[FlatNetlist, list[ArityError]]:
    """``flatten_graph``, or with ``arity`` a list, the lenient flattening
    ``compare_graphs`` falls back to, which records each port-arity mismatch
    there and keeps that instance as a black box (see ``_expand``)."""
    leaves: list[FlatComponent] = []
    calls: list[FlatCall] = []
    unresolved: set[str] = set()
    for comp in graph.components:
        _expand(
            comp,
            graph.subckts,
            path=(),
            net_map=None,
            global_nets=graph.global_nets,
            reader=graph.reader,
            out=leaves,
            calls=calls,
            stack=(),
            unresolved=unresolved,
            arity=arity,
        )
    nets = frozenset(net for leaf in leaves for net in leaf.nodes)
    flat = FlatNetlist(
        components=tuple(leaves),
        nets=nets,
        unresolved_subckts=tuple(sorted(unresolved)),
        calls=tuple(calls),
    )
    return flat, [] if arity is None else arity


def _expand(
    comp: Component,
    subckts: dict[str, SubcktDef],
    *,
    path: tuple[str, ...],
    net_map: dict[str, str] | None,
    global_nets: frozenset[str],
    reader: SemanticProfile,
    out: list[FlatComponent],
    calls: list[FlatCall],
    stack: tuple[str, ...],
    unresolved: set[str],
    arity: list[ArityError] | None = None,
) -> None:
    """Recursively expand ``comp`` (in the namespace described by ``net_map``).

    ``net_map`` translates a net name as written inside the current subcircuit
    body to its flattened name; ``None`` at the top level means identity. ``path``
    is the chain of enclosing instance refs (for hierarchical naming and unique
    internal-net keys). ``unresolved`` collects the names of subcircuits an ``X``
    asked for but nothing defined.

    With ``arity`` None a port-arity mismatch raises :class:`PortArityMismatch`
    and a recursive definition raises; with a list, the mismatch is recorded
    there, and the instance, or one that recurses, is kept as a black-box leaf.
    """

    def resolve(net: str) -> str:
        return net if net_map is None else net_map.get(net, net)

    flat_nodes = tuple(resolve(n) for n in comp.nodes)
    model_key = comp.model.lower() if comp.model else None
    is_expandable = comp.type_letter == "X" and model_key is not None and model_key in subckts
    # Recorded before the arity/recursion guards below can clear the flag: those
    # black-box an instance whose definition DOES exist, which is not "unresolved".
    if comp.type_letter == "X" and comp.model and not is_expandable:
        unresolved.add(comp.model)

    if is_expandable:
        assert model_key is not None
        sub = subckts[model_key]
        if len(comp.nodes) != len(sub.ports):
            if arity is None:
                raise PortArityMismatch(
                    "port-arity mismatch: "
                    + _instance_port_detail(
                        comp.ref, tuple(comp.nodes), sub.name, tuple(sub.ports)
                    ),
                    line=comp.line,
                )
            ref = path_text((*path, comp.ref))
            arity.append(
                ArityError(
                    ref=ref,
                    reference_arity=len(comp.nodes),
                    candidate_arity=len(sub.ports),
                    detail=_instance_port_detail(
                        ref, tuple(comp.nodes), comp.model or model_key, tuple(sub.ports)
                    ),
                )
            )
            is_expandable = False
        elif model_key in stack:
            if arity is None:
                chain = " -> ".join((*stack, model_key))
                raise NetlistGraphError(f"recursive .SUBCKT definition: {chain}")
            is_expandable = False

    if not is_expandable:
        out.append(
            FlatComponent(
                ref=path_text((*path, comp.ref)),
                type_letter=comp.type_letter,
                nodes=flat_nodes,
                model=comp.model,
                value=comp.value,
                params=comp.params,
            )
        )
        return

    assert model_key is not None
    sub = subckts[model_key]
    child_path = (*path, comp.ref)
    calls.append(FlatCall(path_text(child_path), model_key, _passed(sub, comp.params)))
    child_map = _build_child_net_map(sub, flat_nodes, path_text(child_path), global_nets, reader)
    for child in sub.components:
        _expand(
            child,
            subckts,
            path=child_path,
            net_map=child_map,
            global_nets=global_nets,
            reader=reader,
            out=out,
            calls=calls,
            stack=(*stack, model_key),
            unresolved=unresolved,
            arity=arity,
        )


def _build_child_net_map(
    sub: SubcktDef,
    bound_ports: tuple[str, ...],
    prefix: str,
    global_nets: frozenset[str],
    reader: SemanticProfile,
) -> dict[str, str]:
    """Map every net named inside ``sub``'s body to its flattened name, by
    what the name stands for there (``deck_nodes``).

    Ground stays ``0`` and a ``.global`` name stays itself; a port maps to the
    caller's already-flattened connection net; everything else is an internal
    net, given a unique ``<prefix>/<net>`` name (prefix lowercased to stay
    consistent with canonical net names).
    """
    key = f"{prefix.lower()}/"
    bound = {port.casefold(): bound_ports[i] for i, port in enumerate(sub.ports)}
    net_map: dict[str, str] = {}
    for child in sub.components:
        for net in child.nodes:
            if net in net_map:
                continue
            kind = settled_node_kind(
                net,
                ports=sub.ports,
                global_names=global_nets,
                simulator=reader.simulator,
                ngbehavior=reader.ngbehavior,
            )
            if kind == "ground":
                net_map[net] = _GROUND
            elif kind == "global":
                net_map[net] = net
            elif kind == "port":
                net_map[net] = bound[net.casefold()]
            else:
                net_map[net] = f"{key}{net}"
    return net_map


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_graphs(
    reference: NetlistGraph | str | Path,
    candidate: NetlistGraph | str | Path,
    *,
    reader: SemanticProfile | None,
    anchors: list[str] | tuple[str, ...] | frozenset[str] | None = None,
    rtol: float = SAME_VALUE_RTOL,
) -> GraphComparison:
    """Compare two netlists for connectivity equivalence.

    ``reference`` and ``candidate`` are :class:`NetlistGraph` instances or things
    :func:`parse_netlist_graph` accepts (path or text); each is flattened first.

    ``anchors`` are named nets (ports, rails, outputs, measurement nets) that
    must map *by name*: an anchor may not silently land on a different structural
    position between the two sides. Ground is always an implicit anchor.
    ``reader`` is who reads a side given as a path or text
    (``parse_netlist_graph``); a graph carries its own, and two graphs take
    None.
    ``rtol`` is the relative tolerance for numeric value/param comparison.

    Returns a :class:`GraphComparison`. Never a bare bool.
    """
    ref = _coerce_capturing_arity(reference, reader)
    cand = _coerce_capturing_arity(candidate, reader)

    result = GraphComparison(equivalent=False, structurally_equivalent=False)
    result.arity_errors.extend(replace(e, side="reference") for e in ref.arity_errors)
    result.arity_errors.extend(replace(e, side="candidate") for e in cand.arity_errors)

    ref_flat = ref.flat
    cand_flat = cand.flat

    anchor_set = {canon_net(a, ref.graph.reader) for a in (anchors or ())}
    anchor_set.add(_GROUND)

    # --- structural (wiring) verdict + anchor placement, via 1-WL colors ------
    colors = _wl_refine(ref_flat, cand_flat)
    result.structurally_equivalent = _histograms_match(ref_flat, cand_flat, colors)
    result.anchor_violations.extend(_check_anchors(ref_flat, cand_flat, colors, anchor_set))

    # --- per-component diffs, by reference-name matching ----------------------
    ref_by_ref = {canon_ref(c.ref): c for c in ref_flat.components}
    cand_by_ref = {canon_ref(c.ref): c for c in cand_flat.components}
    cand_by_ref, paired = rekey_across_added_x(ref_by_ref, cand_by_ref)
    result.renamed = _renamed_instances(ref_by_ref, cand_by_ref, paired)

    for key, rc in ref_by_ref.items():
        if key not in cand_by_ref:
            result.removed.append(
                ComponentDelta(ref=rc.ref, type_letter=rc.type_letter, detail=_describe(rc))
            )
    for key, cc in cand_by_ref.items():
        if key not in ref_by_ref:
            result.added.append(
                ComponentDelta(ref=cc.ref, type_letter=cc.type_letter, detail=_describe(cc))
            )

    for key in ref_by_ref.keys() & cand_by_ref.keys():
        rc, cc = ref_by_ref[key], cand_by_ref[key]
        _diff_matched_pair(rc, cc, rtol, result)
    # A subcircuit instance is no leaf, but its own parameters set what its
    # body's values read, so an instance both sides have compares them too.
    result.calls = paired_calls(ref_flat, cand_flat)
    for ref_call, cand_call in result.calls:
        _diff_params(ref_call.ref, ref_call.params, cand_call.params, rtol, result)

    # --- named node-partition mismatches, via reference-keyed propagation -----
    result.node_partition_mismatches.extend(
        _propagate_partition(ref_by_ref, cand_by_ref, anchor_set)
    )

    # --- fact-level context: black boxes and duplicate definitions ------------
    result.unresolved_subckts.extend(_unresolved_records(ref, cand))
    for outcome in (ref, cand):
        for dup in outcome.graph.duplicate_subckts:
            if dup not in result.duplicate_subckts:
                result.duplicate_subckts.append(dup)
    for side, outcome in (("reference", ref), ("candidate", cand)):
        for subckt, comp in _cards_in_use(outcome.graph):
            for key, values in copies_by_name(comp.params).items():
                if len(values) > 1:
                    result.duplicate_params.append(
                        DuplicateParam(comp.ref, subckt, side, key, tuple(values))
                    )

    result.equivalent = (
        result.structurally_equivalent
        and not result.added
        and not result.removed
        and not result.retyped
        and not result.value_mismatches
        and not result.param_mismatches
        and not result.node_partition_mismatches
        and not result.anchor_violations
        and not result.arity_errors
    )
    return result


_V = TypeVar("_V")


def rekey_across_added_x(
    reference: Mapping[str, object], candidate: Mapping[str, _V]
) -> tuple[dict[str, _V], list[str]]:
    """``candidate`` with each key that matches a reference key only across
    LTspice's added ``X`` re-keyed to that reference key, and those keys.

    LTspice exports a subcircuit symbol's instance with an ``X`` before its
    name whatever the name is: ``Xe`` is ``X§Xe`` from LTspice 24 on and ``XXe``
    from XVII, where a netlist written by hand names it ``Xe``. Keys are
    :func:`canon_ref` keys, hierarchical ones included. Only keys left unmatched
    by name on both sides are paired, and only one to one, so a netlist that
    really holds both ``Xe`` and ``XXe`` is compared by name. The re-keyed
    reference keys come back sorted.
    """

    def unmatched(keys: Iterable[str], others: Mapping[str, object]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for key in keys:
            if key not in others:
                grouped.setdefault(without_added_x(key), []).append(key)
        return grouped

    by_reference = unmatched(reference, candidate)
    rekeyed = dict(candidate)
    paired: list[str] = []
    for stripped, cand_keys in unmatched(candidate, reference).items():
        ref_keys = by_reference.get(stripped, [])
        if len(cand_keys) == 1 and len(ref_keys) == 1:
            rekeyed[ref_keys[0]] = rekeyed.pop(cand_keys[0])
            paired.append(ref_keys[0])
    return rekeyed, sorted(paired)


def _cards_in_use(graph: NetlistGraph) -> list[tuple[str | None, Component]]:
    """Each card the top level reaches, once, with the subcircuit it is in
    (``None`` at the top level): the top level's, then each subcircuit's an
    ``X`` instance calls, in the order first called."""
    cards: list[tuple[str | None, Component]] = [(None, comp) for comp in graph.components]
    seen: set[str] = set()
    for _subckt, comp in cards:
        key = comp.model.lower() if comp.type_letter == "X" and comp.model else None
        if key is None or key in seen or key not in graph.subckts:
            continue
        seen.add(key)
        sub = graph.subckts[key]
        cards.extend((sub.name, inner) for inner in sub.components)
    return cards


def _renamed_instances(
    ref_by_ref: dict[str, FlatComponent], cand_by_ref: dict[str, FlatComponent], paired: list[str]
) -> list[RenamedInstance]:
    """Each instance a paired leaf's path renames, named once."""
    renamed: dict[tuple[str, str], RenamedInstance] = {}
    for key in paired:
        segments = zip(
            ref_by_ref[key].ref.split("."), cand_by_ref[key].ref.split("."), strict=False
        )
        for ref_seg, cand_seg in segments:
            if canon_ref(ref_seg) != canon_ref(cand_seg):
                renamed.setdefault((ref_seg, cand_seg), RenamedInstance(ref_seg, cand_seg))
    return sorted(renamed.values(), key=lambda d: canon_ref(d.reference_ref))


def _unresolved_records(ref: _FlattenOutcome, cand: _FlattenOutcome) -> list[UnresolvedSubckt]:
    """One record per subcircuit left black-boxed, saying which side(s) and why.

    Each record carries that side's unusable include targets, so a caller can
    tell a model whose library was never referenced (no missing includes) from a
    definition file that failed to load (missing includes listed).
    """
    ref_names = {n.lower(): n for n in ref.flat.unresolved_subckts}
    cand_names = {n.lower(): n for n in cand.flat.unresolved_subckts}
    ref_missing = tuple(m.target for m in ref.graph.missing_includes)
    cand_missing = tuple(m.target for m in cand.graph.missing_includes)

    records: list[UnresolvedSubckt] = []
    for key in sorted(ref_names.keys() | cand_names.keys()):
        in_ref, in_cand = key in ref_names, key in cand_names
        if in_ref and in_cand:
            side, missing = "both", tuple(dict.fromkeys(ref_missing + cand_missing))
        elif in_ref:
            side, missing = "reference", ref_missing
        else:
            side, missing = "candidate", cand_missing
        records.append(
            UnresolvedSubckt(
                name=ref_names.get(key) or cand_names[key],
                side=side,
                missing_includes=missing,
            )
        )
    return records


@dataclass
class _FlattenOutcome:
    flat: FlatNetlist
    arity_errors: list[ArityError]
    graph: NetlistGraph


def _coerce_capturing_arity(
    src: NetlistGraph | str | Path, reader: SemanticProfile | None
) -> _FlattenOutcome:
    """Flatten ``src``, turning a flattening arity error into a recorded fact.

    A malformed *card* still raises (that is a parse failure); only the
    connectivity-level arity mismatch is captured so ``compare_graphs`` can
    report it alongside the other differences instead of aborting.
    """
    if isinstance(src, NetlistGraph):
        graph = src
    elif reader is None:
        raise TypeError("a netlist given as a path or text needs its reader")
    else:
        graph = parse_netlist_graph(src, reader=reader)
    try:
        flat, arity = flatten_graph(graph), []
    except PortArityMismatch:
        # Re-flatten skipping the offending instances so the rest still compares.
        flat, arity = _flatten(graph, arity=[])
    return _FlattenOutcome(flat=_with_bjt_substrate(flat), arity_errors=arity, graph=graph)


def _with_bjt_substrate(flat: FlatNetlist) -> FlatNetlist:
    """Give every three-node ``Q`` card its substrate at ground.

    Both LTspice builds run ``Q1 c b e QN`` as ``Q1 c b e 0 QN``, the card the
    export writes for a three-pin transistor symbol, so the two compare equal.
    Applied after flattening, where ``0`` inside a subcircuit is already the
    global ground it is to the simulator.
    """
    if not any(c.type_letter == "Q" and len(c.nodes) == 3 for c in flat.components):
        return flat
    components = tuple(
        replace(c, nodes=(*c.nodes, _GROUND)) if c.type_letter == "Q" and len(c.nodes) == 3 else c
        for c in flat.components
    )
    return replace(flat, components=components, nets=flat.nets | {_GROUND})


def _describe(comp: FlatComponent) -> str:
    """Short human description of a leaf component for add/remove reporting."""
    ident = comp.model or comp.value or ""
    return f"{comp.type_letter} {ident}".strip()


def _type_signature(comp: FlatComponent) -> str:
    """Type identity for retype detection: element letter plus model when any."""
    return f"{comp.type_letter}:{comp.model.lower()}" if comp.model else comp.type_letter


# Why a three-node M card does not match the four-node card an export writes,
# and what to write instead. Recorded: deck/mos_three_node,
# deck/vdmos_three_node and deck/vdmos_bulk_elsewhere.
_THREE_NODE_MOSFET = (
    " A three-node M card is not the card an export writes for a three-pin MOSFET"
    " symbol, which puts the bulk at the source: with a level-1 model LTspice 26"
    " refuses it and LTspice XVII puts the bulk at ground. Write the bulk as the"
    " fourth node, at the source to match the export. A VDMOS card runs the same"
    " either way, since LTspice requires its bulk to be its source."
)


def _diff_matched_pair(
    rc: FlatComponent, cc: FlatComponent, rtol: float, result: GraphComparison
) -> None:
    """Populate retype / arity / value / param diffs for one matched pair."""
    if len(rc.nodes) != len(cc.nodes):
        detail = (
            f"{rc.ref} has {len(rc.nodes)} terminal(s) in the reference "
            f"({', '.join(rc.nodes)}) and {len(cc.nodes)} in the candidate "
            f"({', '.join(cc.nodes)})."
        )
        if rc.type_letter == "M" and {len(rc.nodes), len(cc.nodes)} == {3, 4}:
            detail += _THREE_NODE_MOSFET
        result.arity_errors.append(
            ArityError(
                ref=rc.ref,
                reference_arity=len(rc.nodes),
                candidate_arity=len(cc.nodes),
                detail=detail,
            )
        )
    if _type_signature(rc).lower() != _type_signature(cc).lower():
        result.retyped.append(
            RetypeDiff(
                ref=rc.ref,
                reference_type=_type_signature(rc),
                candidate_type=_type_signature(cc),
            )
        )
    if not values_equal(rc.value, cc.value, rtol):
        result.value_mismatches.append(
            ValueDiff(ref=rc.ref, reference_value=rc.value, candidate_value=cc.value)
        )
    _diff_params(rc.ref, rc.params, cc.params, rtol, result)


def _passed(
    definition: SubcktDef, given: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """The parameters an instance runs ``definition`` with, as LTspice reads
    an instance: each it gives, every copy kept, else the definition's header
    default (recorded: ``deck/params_keyword``, ``deck/subckt_param_precedence``;
    ``hierarchy`` reads the same precedence). An instance that gives the
    default's value and one that leaves it out run alike."""
    named = {key.casefold() for key, _ in given}
    defaults = tuple(
        (key, value) for key, value in definition.param_defaults if key.casefold() not in named
    )
    return (*defaults, *given)


def _diff_params(
    ref: str,
    reference: tuple[tuple[str, str], ...],
    candidate: tuple[tuple[str, str], ...],
    rtol: float,
    result: GraphComparison,
) -> None:
    """A ``param_mismatches`` row for each parameter of ``ref`` the two sides
    give otherwise (``_disagreement``)."""
    rparams, cparams = copies_by_name(reference), copies_by_name(candidate)
    for pkey in sorted(rparams.keys() | cparams.keys()):
        differ = _disagreement(rparams.get(pkey, []), cparams.get(pkey, []), rtol)
        if differ is not None:
            result.param_mismatches.append(
                ParamDiff(ref=ref, key=pkey, reference_value=differ[0], candidate_value=differ[1])
            )


def paired_calls(
    reference: FlatNetlist, candidate: FlatNetlist
) -> list[tuple[FlatCall, FlatCall]]:
    """The subcircuit instances both sides expand, paired by path as the
    leaves are (``canon_ref``, across the ``X`` LTspice adds), in path order:
    what tells which definition on one side stands for which on the other,
    whatever each is named."""
    ref = {canon_ref(call.ref): call for call in reference.calls}
    cand, _ = rekey_across_added_x(ref, {canon_ref(call.ref): call for call in candidate.calls})
    return [(ref[key], cand[key]) for key in sorted(ref.keys() & cand.keys())]


def _disagreement(
    rvals: list[str], cvals: list[str], rtol: float
) -> tuple[str | None, str | None] | None:
    """``None`` when a parameter reads the same on both sides whichever copy is
    read, else a reference copy and a candidate copy for the mismatch row
    (``None`` on a side that does not give it).

    One copy a side is the plain comparison. With more, the sides agree when
    they give the same copies in the same order, or when every copy on both
    sides is the same value; anything else depends on which copy a simulator
    takes, which this comparison does not model.
    """
    if rvals and cvals:
        if len(rvals) == len(cvals) and all(
            values_equal(r, c, rtol) for r, c in zip(rvals, cvals, strict=True)
        ):
            return None
        if all(values_equal(rvals[0], v, rtol) for v in (*rvals[1:], *cvals)):
            return None
    first = (rvals[0] if rvals else None, cvals[0] if cvals else None)
    # Within tolerance every cross pair can agree while two copies on one side
    # do not; the duplicate_params fact lists those.
    return next(((r, c) for r in rvals for c in cvals if not values_equal(r, c, rtol)), first)


# ---------------------------------------------------------------------------
# 1-WL color refinement over the bipartite component/net graph
# ---------------------------------------------------------------------------


def _wl_refine(ref: FlatNetlist, cand: FlatNetlist) -> dict[tuple[str, str], int]:
    """Jointly refine colors over both graphs' union so colors are comparable.

    Node ids are ``(side, key)`` where side is ``"rc"``/``"cc"`` for a component
    (key = its ref) and ``"rn"``/``"cn"`` for a net (key = its name). Ground is
    pinned to its own initial color (an implicit anchor); anchor *labels* are
    intentionally not seeded here — this pass judges wiring only.
    """
    # Incidence, both directions (undirected bipartite): each pin contributes a
    # (role, neighbor-id) to the component and to the net.
    pins: dict[tuple[str, str], list[tuple[int, tuple[str, str]]]] = {}
    init: dict[tuple[str, str], tuple[str, ...]] = {}

    def add_side(flat: FlatNetlist, comp_side: str, net_side: str) -> None:
        for net in flat.nets:
            nid = (net_side, net)
            pins.setdefault(nid, [])
            init[nid] = ("gnd",) if net == _GROUND else ("net",)
        for comp in flat.components:
            cid = (comp_side, canon_ref(comp.ref))
            pins.setdefault(cid, [])
            init[cid] = ("comp", str(len(comp.nodes)))
            for idx, net in enumerate(comp.nodes):
                role = _pin_role(comp.type_letter, idx)
                nid = (net_side, net)
                pins[cid].append((role, nid))
                pins.setdefault(nid, []).append((role, cid))

    add_side(ref, "rc", "rn")
    add_side(cand, "cc", "cn")

    # Intern each round's color signatures into small integers so equal
    # neighborhoods across the two graphs receive equal codes (comparability).
    color: dict[tuple[str, str], int] = {}
    encoder: dict[object, int] = {}
    for nid, sig in init.items():
        color[nid] = encoder.setdefault(("init", sig), len(encoder))

    prev = len(set(color.values()))
    for _ in range(len(pins) + 1):
        round_enc: dict[object, int] = {}
        new: dict[tuple[str, str], int] = {}
        for nid, incident in pins.items():
            neigh = tuple(sorted((role, color[nbr]) for role, nbr in incident))
            new[nid] = round_enc.setdefault((color[nid], neigh), len(round_enc))
        color = new
        if len(round_enc) == prev:
            break
        prev = len(round_enc)
    return color


def _histograms_match(
    ref: FlatNetlist, cand: FlatNetlist, colors: dict[tuple[str, str], int]
) -> bool:
    """1-WL necessary condition: component and net color multisets must match."""
    ref_comp = Counter(colors[("rc", canon_ref(c.ref))] for c in ref.components)
    cand_comp = Counter(colors[("cc", canon_ref(c.ref))] for c in cand.components)
    if ref_comp != cand_comp:
        return False
    ref_net = Counter(colors[("rn", n)] for n in ref.nets)
    cand_net = Counter(colors[("cn", n)] for n in cand.nets)
    return ref_net == cand_net


def _check_anchors(
    ref: FlatNetlist,
    cand: FlatNetlist,
    colors: dict[tuple[str, str], int],
    anchors: set[str],
) -> list[AnchorViolation]:
    """Each anchor must sit on structurally corresponding nets on both sides.

    Uses the wiring-pass colors: if the two graphs are isomorphic, corresponding
    positions share a color, so an anchor whose reference and candidate nets have
    different colors has been moved to a non-corresponding position.
    """
    violations: list[AnchorViolation] = []
    for anchor in sorted(anchors):
        in_ref = anchor in ref.nets
        in_cand = anchor in cand.nets
        if in_ref and in_cand:
            if colors[("rn", anchor)] != colors[("cn", anchor)]:
                violations.append(
                    AnchorViolation(
                        anchor=anchor,
                        detail="anchor net occupies non-corresponding structural positions",
                    )
                )
        elif in_ref or in_cand:
            side = "reference" if in_ref else "candidate"
            violations.append(
                AnchorViolation(
                    anchor=anchor,
                    detail=f"anchor net present only in {side}",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Reference-keyed net-correspondence propagation (names the partition mismatch)
# ---------------------------------------------------------------------------


def _propagate_partition(
    ref_by_ref: dict[str, FlatComponent],
    cand_by_ref: dict[str, FlatComponent],
    anchors: set[str],
) -> list[NodePartitionDiff]:
    """Derive a reference-net -> candidate-net bijection from matched components.

    Anchors (and ground) seed the mapping by name; every matched component then
    forces its terminals' nets to correspond. The mapping must stay bijective, so
    two failure modes are recorded — both naming every net on the collision:

    - **fan-out** — one reference net is pushed onto two candidate nets;
    - **short** — two reference nets are pushed onto one candidate net.

    Requires shared reference names; when refs are disjoint the matched set is
    empty and this returns nothing (the WL verdict still stands).
    """
    fwd: dict[str, str] = {}
    bwd: dict[str, str] = {}
    # fan_out: reference net -> (candidate nets it is forced onto, witnessing pins)
    fan_out: dict[str, tuple[set[str], set[str]]] = {}
    # shorts: candidate net -> (reference nets merged onto it, witnessing pins)
    shorts: dict[str, tuple[set[str], set[str]]] = {}

    def equate(rnet: str, cnet: str, pin_ref: str, pin_idx: int) -> None:
        pin = f"{pin_ref}.{pin_idx + 1}"
        existing = fwd.get(rnet)
        if existing is not None and existing != cnet:
            cand_nets, involved = fan_out.setdefault(rnet, (set(), set()))
            cand_nets.update({existing, cnet})
            involved.add(pin)
            return
        other = bwd.get(cnet)
        if other is not None and other != rnet:
            # Two distinct reference nets want the same candidate net: name both,
            # not just the newcomer, so the report stands on its own.
            ref_nets, involved = shorts.setdefault(cnet, (set(), set()))
            ref_nets.update({other, rnet})
            involved.add(pin)
            return
        fwd[rnet] = cnet
        bwd[cnet] = rnet

    for anchor in anchors:
        fwd.setdefault(anchor, anchor)
        bwd.setdefault(anchor, anchor)

    shared = ref_by_ref.keys() & cand_by_ref.keys()
    # Ordered devices give direct pin-to-pin constraints; symmetric passives are
    # resolved iteratively once one of their two nets is pinned down.
    ordered: list[tuple[FlatComponent, FlatComponent]] = []
    symmetric: list[tuple[FlatComponent, FlatComponent]] = []
    for key in shared:
        rc, cc = ref_by_ref[key], cand_by_ref[key]
        if len(rc.nodes) != len(cc.nodes):
            continue
        if rc.type_letter in _SYMMETRIC_TWO_TERMINAL and len(rc.nodes) == 2:
            symmetric.append((rc, cc))
        else:
            ordered.append((rc, cc))

    for rc, cc in ordered:
        for idx, (rnet, cnet) in enumerate(zip(rc.nodes, cc.nodes, strict=True)):
            equate(rnet, cnet, rc.ref, idx)

    # Fixpoint over symmetric passives. A resistor's two terminals are
    # interchangeable, so the orientation is fixed by whichever terminal is
    # already pinned (by an anchor or an earlier device). Once a terminal is
    # pinned we equate both pins, which BINDS a still-free terminal and VERIFIES
    # an already-mapped one — surfacing a conflict when both ends are anchored
    # but the wiring disagrees (a moved tap). A device with both terminals free
    # is revisited on a later round once a neighbor pins one of its nets.
    changed = True
    while changed:
        changed = False
        for rc, cc in symmetric:
            (ra, rb), (ca, cb) = rc.nodes, cc.nodes
            ra_img, rb_img = fwd.get(ra), fwd.get(rb)
            if ra_img is not None:
                orient_a = ra_img != cb  # ra->ca unless ra is already pinned to cb
            elif rb_img is not None:
                orient_a = rb_img != ca  # rb->cb unless rb is already pinned to ca
            else:
                continue  # both free; wait for a neighbor to pin one net
            pairs = ((ra, ca), (rb, cb)) if orient_a else ((ra, cb), (rb, ca))
            for idx, (rnet, cnet) in enumerate(pairs):
                if fwd.get(rnet) == cnet and bwd.get(cnet) == rnet:
                    continue  # already established, no work
                fresh = rnet not in fwd and cnet not in bwd
                equate(rnet, cnet, rc.ref, idx)
                if fresh and fwd.get(rnet) == cnet:
                    changed = True

    diffs = [
        NodePartitionDiff(
            reference_nets=(rnet,),
            candidate_nets=tuple(sorted(cand_nets)),
            involved=tuple(sorted(involved)),
        )
        for rnet, (cand_nets, involved) in sorted(fan_out.items())
    ]
    diffs += [
        NodePartitionDiff(
            reference_nets=tuple(sorted(ref_nets)),
            candidate_nets=(cnet,),
            involved=tuple(sorted(involved)),
        )
        for cnet, (ref_nets, involved) in sorted(shorts.items())
    ]
    return diffs
