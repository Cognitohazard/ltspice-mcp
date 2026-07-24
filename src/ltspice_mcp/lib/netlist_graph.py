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
depth-bounded and cycle-guarded, with the ``.lib file section`` form taking the
file token). When one name is defined more than once, the FIRST definition in
textual order wins and the losers are recorded — measured behavior, not
assumption: ngspice 42 warns ``redefinition of .subckt X, ignored`` and keeps
whichever definition came first, so an inline block only wins when it precedes
the include (the usual LTspice layout), not because it is inline.
- :func:`compare_graphs` flattens both sides and reports what differs:
  components added / removed / retyped, value and parameter mismatches, node
  partition (connectivity) mismatches, anchor violations, and port-arity
  errors. It never returns a bare bool.

Ground normalization follows LTspice's export convention (verified against real
exports: LTspice emits ``0`` for ground). ``0`` and ``GND`` — and the common
``GND!`` variant — are treated as the same net. All net and reference matching
is case-insensitive, per SPICE.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.spice_lex import (
    LexResult,
    SpiceCard,
    SpiceLexError,
    cards_from_path,
    iter_by_kind,
    lex,
)
from ltspice_mcp.lib.spice_lex_views import InstanceLine, SubcktCard

# An include-path resolver: given the filesystem path an ``.include``/``.lib``
# resolved to (base directory joined with the target, or an absolute target),
# it returns the path to actually open, or ``None`` to deny the open. The engine
# never reads a denied path — the callback is the sandbox seam the tool layer
# wires to ``safe_path`` so an in-deck include cannot escape the allowed roots.
IncludeResolver = Callable[[Path], "Path | None"]

# LTspice writes ground as ``0``; ``GND``/``GND!`` are common hand-written
# aliases. All fold to ``0`` so a schematic drawn with a GND flag and a netlist
# written with node ``0`` compare equal.
GROUND_ALIASES: frozenset[str] = frozenset({"0", "gnd", "gnd!"})
_GROUND = "0"

# Passive two-terminal devices whose terminals are electrically interchangeable.
# Their pins share one role so a resistor authored "backwards" (b a instead of
# a b) is still recognized as the same connection. Everything else keeps pin
# order: a MOSFET's D/G/S/B, a source's +/-, a diode's anode/cathode all matter.
_SYMMETRIC_TWO_TERMINAL: frozenset[str] = frozenset({"R", "C", "L"})

# Micro sign variants LTspice emits (U+00B5 MICRO SIGN) and Greek mu (U+03BC),
# neither of which the value parser's ASCII-suffix regex accepts. Normalized to
# ``u`` before parsing so ``2.2µ`` compares equal to ``2.2u`` / ``2.2e-6``.
_MICRO_CHARS = ("µ", "μ")

# LTspice prefixes schematic-derived subcircuit instance names with a private
# section sign (``X§RB``). Stripped so an export compares equal to a
# hand-written ``XRB``.
_LTSPICE_INSTANCE_MARKER = "§"

# Directives that pull another file into the deck. Semantics mirror
# ``sim_runner.deck_requests_raw`` / ``_include_target`` (kept in step with that
# module deliberately rather than imported — this stays a leaf module): a target
# is resolved against the INCLUDING file's own directory, the walk is
# depth-bounded and cycle-guarded, and the ``.lib file section`` form takes the
# file token (the section name is irrelevant to a subcircuit scan).
_INCLUDE_DIRECTIVES: frozenset[str] = frozenset({".include", ".inc", ".lib"})
_MAX_INCLUDE_DEPTH = 3


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


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One element instance, projected out of a lexed card.

    ``ref`` has the LTspice ``§`` marker stripped but otherwise preserves source
    spelling; matching is case-insensitive (:func:`canon_ref`). ``nodes`` are
    canonicalized net names (ground folded to ``0``, lowercased). ``type_letter``
    is the uppercased element prefix. ``model`` is the model / subcircuit name
    for M/Q/J/X (and controlling-source name for F/H); ``value`` is the passive
    value or source spec for R/C/L/V/I. ``params`` are the ``key=value`` tokens.
    """

    ref: str
    type_letter: str
    nodes: tuple[str, ...]
    model: str | None
    value: str | None
    params: tuple[tuple[str, str], ...]
    line: int = 0

    @property
    def params_dict(self) -> dict[str, str]:
        return dict(self.params)


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


@dataclass(frozen=True)
class MissingInclude:
    """An ``.include``/``.lib`` target that could not contribute definitions.

    ``target`` is the file token exactly as the deck wrote it; ``reason`` says
    why it yielded nothing (not found, unreadable, past the depth bound, or no
    base directory because the netlist was parsed from bare text).
    """

    target: str
    reason: str


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
    warnings: list[str] = field(default_factory=list)
    missing_includes: tuple[MissingInclude, ...] = ()
    duplicate_subckts: tuple[DuplicateSubckt, ...] = ()


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

    @property
    def params_dict(self) -> dict[str, str]:
        return dict(self.params)


@dataclass(frozen=True)
class FlatNetlist:
    """A fully flattened netlist: leaf components and the set of nets.

    ``unresolved_subckts`` names the subcircuits an ``X`` instance asked for but
    that no inline or included definition supplied — each left as a black box.
    """

    components: tuple[FlatComponent, ...]
    nets: frozenset[str]
    unresolved_subckts: tuple[str, ...] = ()


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
    """A matched component whose terminal count differs between the two sides."""

    ref: str
    reference_arity: int
    candidate_arity: int


@dataclass
class GraphComparison:
    """The structured result of :func:`compare_graphs`.

    ``equivalent`` is the single-glance verdict: structurally isomorphic *and*
    no component, value, parameter, anchor, or arity difference.
    ``structurally_equivalent`` is the wiring-only verdict (the 1-WL histogram
    check, ground pinned, anchor labels ignored) — it can be True while
    ``equivalent`` is False when only values or anchor placement differ.
    """

    equivalent: bool
    structurally_equivalent: bool
    added: list[ComponentDelta] = field(default_factory=list)
    removed: list[ComponentDelta] = field(default_factory=list)
    retyped: list[RetypeDiff] = field(default_factory=list)
    value_mismatches: list[ValueDiff] = field(default_factory=list)
    param_mismatches: list[ParamDiff] = field(default_factory=list)
    node_partition_mismatches: list[NodePartitionDiff] = field(default_factory=list)
    anchor_violations: list[AnchorViolation] = field(default_factory=list)
    arity_errors: list[ArityError] = field(default_factory=list)
    # Fact-level context, not differences: a black-boxed subcircuit and a
    # duplicate definition are both legitimate, so neither flips ``equivalent``.
    unresolved_subckts: list[UnresolvedSubckt] = field(default_factory=list)
    duplicate_subckts: list[DuplicateSubckt] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Project to a plain-dict payload (JSON-friendly)."""
        return {
            "equivalent": self.equivalent,
            "structurally_equivalent": self.structurally_equivalent,
            "added": [vars(d) for d in self.added],
            "removed": [vars(d) for d in self.removed],
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
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def canon_net(name: str) -> str:
    """Canonicalize a net name: strip, lowercase, fold ground aliases to ``0``."""
    n = name.strip().lower()
    return _GROUND if n in GROUND_ALIASES else n


def canon_ref(ref: str) -> str:
    """Canonicalize a reference for matching: drop the ``§`` marker, lowercase."""
    return ref.replace(_LTSPICE_INSTANCE_MARKER, "").strip().lower()


def _strip_marker(ref: str) -> str:
    """Drop the LTspice ``§`` marker while preserving reference casing."""
    return ref.replace(_LTSPICE_INSTANCE_MARKER, "")


def _normalize_value_text(text: str) -> str:
    """Replace micro-sign variants with ``u`` (for parsing and string compare)."""
    out = text
    for ch in _MICRO_CHARS:
        out = out.replace(ch, "u")
    return out


def values_equal(a: str | None, b: str | None, rtol: float) -> bool:
    """Compare two value/param strings, numeric within ``rtol`` else textually.

    ``1k`` == ``1000`` == ``1.0k``; ``4.7µ`` == ``4.7u`` == ``4.7e-6``. When
    either side is not a parseable SPICE number (a model name, a ``PULSE(...)``
    spec), fall back to a case-insensitive, whitespace-collapsed string compare.
    """
    if a is None or b is None:
        return a == b
    na, nb = _normalize_value_text(a), _normalize_value_text(b)
    try:
        va, vb = parse_spice_value(na), parse_spice_value(nb)
    except ValueError:
        return " ".join(na.lower().split()) == " ".join(nb.lower().split())
    if va == vb:
        return True
    scale = max(abs(va), abs(vb))
    if scale == 0.0:
        return True
    return abs(va - vb) <= rtol * scale


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
    """Lex a netlist given as a filesystem path or as raw text.

    A :class:`~pathlib.Path` is always read as a file. A ``str`` is treated as
    raw netlist text when it spans multiple lines or is long; otherwise, if it
    names an existing file, it is read — else it is lexed as text. The second
    element is the directory includes resolve against (``None`` for bare text).
    """
    if isinstance(path_or_text, Path):
        return cards_from_path(path_or_text), path_or_text.parent
    text = path_or_text
    if "\n" not in text and len(text) <= 400:
        try:
            candidate = Path(text)
            if candidate.is_file():
                return cards_from_path(candidate), candidate.parent
        except OSError:
            pass
    return lex(text), None


# Lex warnings that indicate the netlist structure itself is broken (as opposed
# to benign round-trip notes). Any of these means the derived graph would be
# wrong, so parsing raises rather than return a partial graph.
_STRUCTURAL_WARNING_MARKERS = ("unclosed", ".ends", "no preceding card")


def parse_netlist_graph(
    path_or_text: str | Path,
    *,
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
    including file's own directory, depth-bounded and cycle-guarded). When a name
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
    """
    result, source_dir = _load_cards(path_or_text)
    cards = result.cards

    structural = [
        w
        for w in result.warnings
        if any(marker in w.lower() for marker in _STRUCTURAL_WARNING_MARKERS)
    ]
    if structural:
        raise NetlistGraphError("malformed netlist: " + "; ".join(structural))

    registry = _SubcktRegistry()
    _collect_subckts(
        cards,
        base_dir=base_dir if base_dir is not None else source_dir,
        depth=0,
        registry=registry,
        seen=set(),
        source="<inline>",
        strict=True,
        include_resolver=include_resolver,
    )

    components = [_component_from_card(card) for card in iter_by_kind(cards, "instance", scope=())]
    return NetlistGraph(
        components=components,
        subckts=registry.subckts,
        warnings=list(result.warnings),
        missing_includes=tuple(registry.missing),
        duplicate_subckts=registry.duplicate_records(),
    )


def _include_target(rest: str) -> str | None:
    """The filename an ``.include``/``.inc``/``.lib`` argument points at.

    ``rest`` is the directive body past its head. A quoted path is taken whole
    (SPICE allows spaces inside quotes); an unquoted argument is its first
    whitespace-delimited token, which for the ``.lib file section`` form is the
    file (the section name is dropped). Returns None when no target is present.
    """
    rest = rest.strip()
    if not rest:
        return None
    if rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        return rest[1:end] if end != -1 else None
    return rest.split(None, 1)[0]


@dataclass
class _SubcktRegistry:
    """Accumulates subcircuit definitions in textual order, first-wins."""

    subckts: dict[str, SubcktDef] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    ignored: dict[str, list[str]] = field(default_factory=dict)
    missing: list[MissingInclude] = field(default_factory=list)

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


def _subckt_def_from(cards: list[SpiceCard], opener: SpiceCard) -> SubcktDef:
    """Project one ``.SUBCKT`` opener (and its body cards) into a definition."""
    view = SubcktCard.from_card(opener)
    body_scope = (*opener.scope, view.name)
    body = [
        _component_from_card(card) for card in iter_by_kind(cards, "instance", scope=body_scope)
    ]
    return SubcktDef(
        name=view.name,
        ports=tuple(view.ports),
        components=tuple(body),
        param_defaults=tuple(view.param_defaults.items()),
    )


def _collect_subckts(
    cards: list[SpiceCard],
    *,
    base_dir: Path | None,
    depth: int,
    registry: _SubcktRegistry,
    seen: set[Path],
    source: str,
    strict: bool,
    include_resolver: IncludeResolver | None = None,
) -> None:
    """Walk ``cards`` in textual order, registering definitions and following includes.

    Order is what makes first-wins precedence correct: an include contributes its
    definitions at the position its directive appears, exactly as textual
    substitution would. Cards past a top-level ``.END`` are inert and skipped.
    With ``strict`` the caller's own malformed card raises; for an included file
    it is downgraded to a :class:`MissingInclude` record so a third-party library
    cannot break the comparison.
    """
    for card in cards:
        if card.trailing:
            continue
        if card.kind == "subckt":
            try:
                registry.register(_subckt_def_from(cards, card), source)
            except (SpiceLexError, NetlistGraphError) as exc:
                if strict:
                    raise NetlistGraphError(
                        f"malformed .SUBCKT: {exc}",
                        line=card.line_start,
                        card="".join(card.raw_lines).strip(),
                    ) from exc
                registry.missing.append(
                    MissingInclude(target=source, reason=f"malformed .SUBCKT: {exc}")
                )
            continue
        if card.kind != "directive":
            continue
        parts = card.body.split(None, 1)
        if len(parts) < 2 or parts[0].lower() not in _INCLUDE_DIRECTIVES:
            continue
        target = _include_target(parts[1])
        if target is None:
            continue
        _follow_include(
            target,
            base_dir=base_dir,
            depth=depth,
            registry=registry,
            seen=seen,
            include_resolver=include_resolver,
        )


def _follow_include(
    target: str,
    *,
    base_dir: Path | None,
    depth: int,
    registry: _SubcktRegistry,
    seen: set[Path],
    include_resolver: IncludeResolver | None = None,
) -> None:
    """Load one include target's definitions, or record why it contributed none."""
    if base_dir is None:
        registry.missing.append(
            MissingInclude(target, "no base directory (netlist parsed from text)")
        )
        return
    if depth >= _MAX_INCLUDE_DEPTH:
        registry.missing.append(
            MissingInclude(target, f"include depth limit ({_MAX_INCLUDE_DEPTH}) reached")
        )
        return
    raw = Path(target)
    path = raw if raw.is_absolute() else base_dir / raw
    if include_resolver is not None:
        # Gate the open BEFORE any read: a denied include contributes nothing and
        # its bytes are never touched (the sandbox property U6/the tool layer
        # relies on). The resolver may also redirect to a canonical safe path.
        approved = include_resolver(path)
        if approved is None:
            registry.missing.append(MissingInclude(target, "path denied by include resolver"))
            return
        path = approved
    try:
        key = path.resolve()
    except OSError:
        registry.missing.append(MissingInclude(target, "unresolvable path"))
        return
    if key in seen:
        return  # already contributed; revisiting is how an include cycle stops
    seen.add(key)
    try:
        included = cards_from_path(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        registry.missing.append(MissingInclude(target, f"unreadable: {exc}"))
        return
    _collect_subckts(
        included.cards,
        base_dir=path.parent,
        depth=depth + 1,
        registry=registry,
        seen=seen,
        source=str(path),
        strict=False,
        include_resolver=include_resolver,
    )


def _component_from_card(card: object) -> Component:
    """Build a :class:`Component` from an instance card via the typed view."""
    try:
        line = InstanceLine.from_card(card)  # type: ignore[arg-type]
    except SpiceLexError as exc:
        raw = "".join(getattr(card, "raw_lines", [])).strip()
        raise NetlistGraphError(
            f"malformed instance: {exc}",
            line=getattr(card, "line_start", None),
            card=raw,
        ) from exc
    ref = _strip_marker(line.ref)
    return Component(
        ref=ref,
        type_letter=ref[:1].upper(),
        nodes=tuple(canon_net(n) for n in line.nodes),
        model=line.model,
        value=line.value,
        params=tuple(line.params.items()),
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
    leaves: list[FlatComponent] = []
    unresolved: set[str] = set()
    for comp in graph.components:
        _expand(
            comp,
            graph.subckts,
            path=(),
            net_map=None,
            out=leaves,
            stack=(),
            unresolved=unresolved,
        )
    nets = frozenset(net for leaf in leaves for net in leaf.nodes)
    return FlatNetlist(
        components=tuple(leaves),
        nets=nets,
        unresolved_subckts=tuple(sorted(unresolved)),
    )


def _expand(
    comp: Component,
    subckts: dict[str, SubcktDef],
    *,
    path: tuple[str, ...],
    net_map: dict[str, str] | None,
    out: list[FlatComponent],
    stack: tuple[str, ...],
    unresolved: set[str],
) -> None:
    """Recursively expand ``comp`` (in the namespace described by ``net_map``).

    ``net_map`` translates a net name as written inside the current subcircuit
    body to its flattened name; ``None`` at the top level means identity. ``path``
    is the chain of enclosing instance refs (for hierarchical naming and unique
    internal-net keys). ``unresolved`` collects the names of subcircuits an ``X``
    asked for but nothing defined.
    """

    def resolve(net: str) -> str:
        if net_map is None:
            return net
        return net_map.get(net, net)

    flat_nodes = tuple(resolve(n) for n in comp.nodes)
    model_key = comp.model.lower() if comp.model else None

    is_expandable = comp.type_letter == "X" and model_key is not None and model_key in subckts
    if not is_expandable:
        if comp.type_letter == "X" and comp.model:
            unresolved.add(comp.model)
        out.append(
            FlatComponent(
                ref=".".join((*path, comp.ref)),
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
    if len(comp.nodes) != len(sub.ports):
        raise NetlistGraphError(
            f"port-arity mismatch: instance {comp.ref} connects {len(comp.nodes)} node(s) "
            f"but .SUBCKT {sub.name} declares {len(sub.ports)} port(s)",
            line=comp.line,
        )
    if model_key in stack:
        chain = " -> ".join((*stack, model_key))
        raise NetlistGraphError(f"recursive .SUBCKT definition: {chain}")

    child_path = (*path, comp.ref)
    child_prefix = ".".join(child_path)
    child_map = _build_child_net_map(sub, flat_nodes, child_prefix)
    for child in sub.components:
        _expand(
            child,
            subckts,
            path=child_path,
            net_map=child_map,
            out=out,
            stack=(*stack, model_key),
            unresolved=unresolved,
        )


def _build_child_net_map(
    sub: SubcktDef, bound_ports: tuple[str, ...], prefix: str
) -> dict[str, str]:
    """Map every net named inside ``sub``'s body to its flattened name.

    Ports map to the caller's already-flattened connection nets; ground stays
    ``0``; everything else is an internal net, given a unique ``<prefix>/<net>``
    name (prefix lowercased to stay consistent with canonical net names).
    """
    key = f"{prefix.lower()}/"
    port_bind = {canon_net(port): bound_ports[i] for i, port in enumerate(sub.ports)}
    net_map: dict[str, str] = dict(port_bind)
    net_map[_GROUND] = _GROUND
    for child in sub.components:
        for net in child.nodes:
            if net in net_map:
                continue
            net_map[net] = f"{key}{net}"
    return net_map


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_graphs(
    reference: NetlistGraph | str | Path,
    candidate: NetlistGraph | str | Path,
    *,
    anchors: list[str] | tuple[str, ...] | frozenset[str] | None = None,
    rtol: float = 1e-6,
) -> GraphComparison:
    """Compare two netlists for connectivity equivalence.

    ``reference`` and ``candidate`` are :class:`NetlistGraph` instances or things
    :func:`parse_netlist_graph` accepts (path or text); each is flattened first.

    ``anchors`` are named nets (ports, rails, outputs, measurement nets) that
    must map *by name*: an anchor may not silently land on a different structural
    position between the two sides. Ground is always an implicit anchor.
    ``rtol`` is the relative tolerance for numeric value/param comparison.

    Returns a :class:`GraphComparison`. Never a bare bool.
    """
    ref = _coerce_capturing_arity(reference)
    cand = _coerce_capturing_arity(candidate)

    result = GraphComparison(equivalent=False, structurally_equivalent=False)
    result.arity_errors.extend(ref.arity_errors)
    result.arity_errors.extend(cand.arity_errors)

    ref_flat = ref.flat
    cand_flat = cand.flat

    anchor_set = {canon_net(a) for a in (anchors or ())}
    anchor_set.add(_GROUND)

    # --- structural (wiring) verdict + anchor placement, via 1-WL colors ------
    colors = _wl_refine(ref_flat, cand_flat)
    result.structurally_equivalent = _histograms_match(ref_flat, cand_flat, colors)
    result.anchor_violations.extend(_check_anchors(ref_flat, cand_flat, colors, anchor_set))

    # --- per-component diffs, by reference-name matching ----------------------
    ref_by_ref = {canon_ref(c.ref): c for c in ref_flat.components}
    cand_by_ref = {canon_ref(c.ref): c for c in cand_flat.components}

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


def _coerce_capturing_arity(src: NetlistGraph | str | Path) -> _FlattenOutcome:
    """Flatten ``src``, turning a flattening arity error into a recorded fact.

    A malformed *card* still raises (that is a parse failure); only the
    connectivity-level arity mismatch is captured so ``compare_graphs`` can
    report it alongside the other differences instead of aborting.
    """
    graph = src if isinstance(src, NetlistGraph) else parse_netlist_graph(src)
    try:
        return _FlattenOutcome(flat=flatten_graph(graph), arity_errors=[], graph=graph)
    except NetlistGraphError as exc:
        if "port-arity mismatch" not in str(exc):
            raise
        # Re-flatten skipping the offending instances so the rest still compares.
        flat, arity = _flatten_lenient(graph)
        return _FlattenOutcome(flat=flat, arity_errors=arity, graph=graph)


def _flatten_lenient(graph: NetlistGraph) -> tuple[FlatNetlist, list[ArityError]]:
    """Flatten, recording (not raising) port-arity mismatches as black boxes."""
    leaves: list[FlatComponent] = []
    arity: list[ArityError] = []
    unresolved: set[str] = set()
    for comp in graph.components:
        _expand_lenient(
            comp,
            graph.subckts,
            path=(),
            net_map=None,
            out=leaves,
            stack=(),
            arity=arity,
            unresolved=unresolved,
        )
    nets = frozenset(net for leaf in leaves for net in leaf.nodes)
    flat = FlatNetlist(
        components=tuple(leaves),
        nets=nets,
        unresolved_subckts=tuple(sorted(unresolved)),
    )
    return flat, arity


def _expand_lenient(
    comp: Component,
    subckts: dict[str, SubcktDef],
    *,
    path: tuple[str, ...],
    net_map: dict[str, str] | None,
    out: list[FlatComponent],
    stack: tuple[str, ...],
    arity: list[ArityError],
    unresolved: set[str],
) -> None:
    """Like :func:`_expand` but records arity mismatches and keeps the instance
    as a black-box leaf instead of raising."""

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
            arity.append(
                ArityError(
                    ref=".".join((*path, comp.ref)),
                    reference_arity=len(comp.nodes),
                    candidate_arity=len(sub.ports),
                )
            )
            is_expandable = False
        elif model_key in stack:
            is_expandable = False

    if not is_expandable:
        out.append(
            FlatComponent(
                ref=".".join((*path, comp.ref)),
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
    child_map = _build_child_net_map(sub, flat_nodes, ".".join(child_path))
    for child in sub.components:
        _expand_lenient(
            child,
            subckts,
            path=child_path,
            net_map=child_map,
            out=out,
            stack=(*stack, model_key),
            arity=arity,
            unresolved=unresolved,
        )


def _describe(comp: FlatComponent) -> str:
    """Short human description of a leaf component for add/remove reporting."""
    ident = comp.model or comp.value or ""
    return f"{comp.type_letter} {ident}".strip()


def _type_signature(comp: FlatComponent) -> str:
    """Type identity for retype detection: element letter plus model when any."""
    return f"{comp.type_letter}:{comp.model.lower()}" if comp.model else comp.type_letter


def _diff_matched_pair(
    rc: FlatComponent, cc: FlatComponent, rtol: float, result: GraphComparison
) -> None:
    """Populate retype / arity / value / param diffs for one matched pair."""
    if len(rc.nodes) != len(cc.nodes):
        result.arity_errors.append(
            ArityError(
                ref=rc.ref,
                reference_arity=len(rc.nodes),
                candidate_arity=len(cc.nodes),
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
    rparams = {k.lower(): v for k, v in rc.params}
    cparams = {k.lower(): v for k, v in cc.params}
    for pkey in sorted(rparams.keys() | cparams.keys()):
        rv, cv = rparams.get(pkey), cparams.get(pkey)
        if not values_equal(rv, cv, rtol):
            result.param_mismatches.append(
                ParamDiff(ref=rc.ref, key=pkey, reference_value=rv, candidate_value=cv)
            )


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
