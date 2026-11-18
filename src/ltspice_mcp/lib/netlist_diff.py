"""Structural comparison of two netlists, element by element in each scope.

What ``verify_circuit``'s ``structural_diff`` mode and the export's
``diff_vs_prior`` report: components added, removed or changed, and directives
added or removed. Both sides are netlists — an ``.asc`` is compared as its LTspice
export — read with the foundation lexer, so a ``+`` continuation belongs to the
card it continues and an inline comment is not part of it. The equivalence mode
reads each directive with the same reader (:func:`parse_directive`,
:func:`directive_delta`), so the two modes read a card the same way.

An element or directive is its scope plus itself: the top level, or the
``.subckt`` definition it sits in, a nested definition a scope of its own
(:func:`read_deck`). A hierarchical sheet's export has a definition per block,
and blocks reuse references (recorded: ``blocks/nested_sheet_beside_both``),
so a reader keyed by bare reference merges them; the source scan in
``tests/test_deck_reading_owners.py`` holds every such key to its scope. Both
modes compare directives scope by scope, so a ``.param`` moved from a body to
the top level is a difference in each. What they compare differs, on purpose.
A structural diff compares the two files as written: every definition in them,
a subcircuit the circuit does not call among them, and an included file's
cards not at all. Equivalence compares the circuit its wiring comparison
resolves (:func:`circuit_directive_delta`): a definition moved into an include
that only defines subcircuits is a difference here and not there, and a
subcircuit the circuit does not call is compared here only. Tests named for
each difference pin them (``tests/test_verify_circuit.py``).

Two rules keep the delta quiet when nothing changed:

* Directives are compared parsed, not as text. Spacing, case, comma separators,
  the micro sign, and the order of ``.model``/``.param``/``.options`` assignments
  are spelling.
* LTspice's netlister adds lines to every export that say nothing about the
  circuit: ``.backanno``, the install's own ``.lib <install>/lib/cmp/standard.*``
  (at a path that differs per machine), and a parameterless default model such as
  ``.model NMOS NMOS`` per device class. The first two are left out of both sides;
  a default model is left out of the compared deck while the baseline declares no
  model of that name, in that scope or one around it — once it does, the two
  disagree about what the model is.

Components are matched by reference within a scope the way the equivalence mode
matches them (``netlist_graph.canon_ref``), and across the ``X`` LTspice puts
before a subcircuit instance's name on export
(``netlist_graph.rekey_across_added_x``), which is listed as a rename and is not
a difference. They are compared on their model or value plus their instance
parameters. Nodes are left out: equivalence is the mode that compares wiring,
and the delta's ``not_compared`` says so. The export's ``diff_vs_prior`` adds
the pins moved to another net, read from that wiring comparison
(:func:`connection_delta`).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path

from ltspice_mcp.lib.deck_nodes import GROUND
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.export_nets import named_by_export
from ltspice_mcp.lib.format import fold_micro_sign
from ltspice_mcp.lib.netlist_graph import (
    DIRECTIVE_CARD_KINDS,
    SYMMETRIC_TWO_TERMINAL,
    FlatCall,
    NetlistGraph,
    canon_ref,
    compare_graphs,
    rekey_across_added_x,
)
from ltspice_mcp.lib.spice_lex import (
    INCLUDE_HEADS,
    Definition,
    DefinitionRule,
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    directive_head,
    find_matching_ends,
    lex_deck,
    settle_definitions,
    subckt_children,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import ModelCard, read_instance

# Directive cards whose assignments are a set: ``.param a=1 b=2`` and
# ``.param b=2 a=1`` define the same thing. (A ``.model`` is read through its
# typed view, whose parameters are a set by construction.)
_UNORDERED_ASSIGNMENTS = frozenset({".param", ".params", ".option", ".options", ".opt"})

# The parameterless default models LTspice's netlister declares for the stock
# device symbols (``.model NMOS NMOS`` and ``.model PMOS PMOS`` for a MOSFET, and
# likewise for the bipolar, JFET and diode symbols).
_DEFAULT_DEVICE_MODELS = frozenset({"nmos", "pmos", "npn", "pnp", "njf", "pjf", "d"})

# The install's own component library, which the netlister appends as
# ``.lib <install>/lib/cmp/standard.<kind>`` for every device class on the sheet.
# A bare ``standard.<kind>`` names the same file through the library search path.
_STANDARD_LIBRARY = re.compile(r"(?:.*[\\/])?cmp[\\/]standard\.\w+|standard\.\w+", re.IGNORECASE)

# Card kinds read as directives: a scope's (``netlist_graph``), whose ngspice
# ``.control`` lines are read apart, and a definition's opener and closer, which
# a structural diff lists. ``end`` is not one: every deck has it, and a deck
# without it still means the same circuit.
_DIRECTIVE_KINDS = (DIRECTIVE_CARD_KINDS - {"control"}) | {"subckt", "ends"}


def comparable(text: str) -> str:
    """The equality form of a signature or directive: SPICE is case-insensitive,
    and the micro signs mean the same ``u`` suffix."""
    return fold_micro_sign(text).casefold()


def instance_signature(card: SpiceCard) -> str:
    """An instance card's model or value, then its parameters, sorted by name.

    Parameters are part of it because an attribute edit on a sheet (a
    SpiceLine's ``w=``, a diode's area) changes exactly those in the export; their
    order on the card carries no meaning. A parameter the card gives twice signs
    both copies, in card order, so an edit that adds a second copy is a change.
    An unreadable card signs as ``<unparseable>``, as every other netlist read
    reports it.
    """
    inst = read_instance(card)
    if inst is None:
        return "<unparseable>"
    head = inst.display_value()
    parts = [head] if head else []
    if inst.model is not None and inst.value:
        parts.append(inst.value)  # what follows the model: a diode area, a switch state
    prefix = inst.ref[:1].upper()
    head_signed = False
    for key, value in sorted(inst.param_pairs(), key=lambda kv: kv[0].casefold()):
        pair = f"{key}={value}"
        if not head_signed and (pair == head or (key.upper() == prefix and value == inst.value)):
            head_signed = True
            continue  # already the head: a B-source's V=..., a keyed R=1k
        parts.append(pair)
    return " ".join(parts)


def _canonical_value(text: str) -> str:
    """A token's text with its insignificant whitespace removed."""
    if text.startswith("(") and text.endswith(")"):
        return "(" + " ".join(_canonical_words(text[1:-1])) + ")"
    if text.startswith("{") and text.endswith("}"):
        return "".join(text.split())
    return text


def _word(tok: Token) -> str:
    if tok.kind is TokenKind.KEY_VALUE:
        return f"{tok.key}={_canonical_value(tok.value or '')}"
    return _canonical_value(tok.text)


def _canonical_words(body: str) -> list[str]:
    """A card body as tokens: spacing, comma separators and the space around
    ``=`` or inside parentheses are spelling, not content."""
    tokens = takewhile(lambda t: t.kind is not TokenKind.COMMENT_TRAIL, tokenize_body(body))
    return [_word(tok) for tok in tokens]


@dataclass(frozen=True)
class Directive:
    """One directive card, parsed for comparison.

    ``text`` is the card as written — continuation lines joined, inline comment
    dropped, whitespace collapsed — and is what a delta reports. ``key`` decides
    equality. ``model`` is the name a ``.model`` card declares, case folded.
    """

    key: str
    text: str
    model: str | None = None

    @property
    def default_model(self) -> bool:
        """Whether this is one of the netlister's parameterless default models."""
        return (
            self.model in _DEFAULT_DEVICE_MODELS
            and self.key == f".model {self.model} {self.model}"
        )


def parse_directive(card: SpiceCard) -> Directive | None:
    """The comparable form of a directive card, or None for export boilerplate."""
    text = " ".join(card.body.split())
    head = text.split(" ", 1)[0].casefold()
    if head == ".backanno":
        return None
    if head == ".lib" and _STANDARD_LIBRARY.fullmatch(text[len(head) :].strip().strip("\"'")):
        return None
    try:
        if card.kind == "model":
            model = ModelCard.from_card(card)
            # Every copy of a parameter given twice, so a second copy is a change.
            params = sorted(
                (f"{k}={_canonical_value(v)}" for k, v in model.param_pairs()), key=str.casefold
            )
            words = [head, model.name, model.type, *params]
            return Directive(comparable(" ".join(words)), text, model=model.name.casefold())
        words = _canonical_words(card.body)
    except SpiceLexError:
        return Directive(comparable(text), text)  # unbalanced: compare the spelling as written
    if head in _UNORDERED_ASSIGNMENTS:
        rest = words[1:]
        assignments = sorted((w for w in rest if "=" in w), key=str.casefold)
        words = words[:1] + [w for w in rest if "=" not in w] + assignments
    return Directive(comparable(" ".join(words)), text)


#: A scope's key: the names of the definitions it is inside, outermost first,
#: case-folded (empty for the top level).
ScopeKey = tuple[str, ...]


@dataclass
class Scope:
    """One scope of a deck as written: the top level, or one ``.subckt``
    definition, nested ones each their own.

    ``names`` are the definitions it is inside as written, outermost first.
    ``components`` maps a reference as the equivalence mode matches it
    (``canon_ref``: LTspice's ``§`` instance marker dropped, case folded) to
    the reference as written and its signature. ``directives`` are its own
    directive cards. ``definitions`` holds the opener and closer of each
    definition in it, by its name case-folded: they are compared with its
    directives unless that name is not compared (``Deck.unread``).
    """

    names: tuple[str, ...] = ()
    components: dict[str, tuple[str, str]] = field(default_factory=dict)
    directives: list[Directive] = field(default_factory=list)
    definitions: dict[str, list[Directive]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """How a delta names the scope: the definitions it is inside, joined
        with ``/`` (``OUTER/INNER``)."""
        return "/".join(self.names)


@dataclass(frozen=True)
class Deck:
    """A netlist as a structural diff reads it: its elements and directives,
    each in its scope.

    ``scopes`` holds the top level under ``()`` and each definition under its
    :data:`ScopeKey`. ``unread`` holds the key and the name as written of
    each definition the file gives more than once in one scope
    (:func:`read_deck`), whose cards are in no scope.
    ``includes`` is whether the file includes another (``.include``, or a
    ``.lib`` other than the install's own), whose cards are not read.
    """

    scopes: dict[ScopeKey, Scope] = field(default_factory=lambda: {(): Scope()})
    unread: tuple[tuple[ScopeKey, str], ...] = ()
    includes: bool = False


#: A rule for :func:`spice_lex.settle_definitions` that keeps no definition
#: of a name given more than once: a structural diff compares files as
#: written and asks only which definitions share a scope and a name.
_AS_WRITTEN = DefinitionRule(
    "a structural diff", "unestablished", "unestablished", "unestablished"
)

#: The name the lexer gives the scope of a ``.subckt`` with no name.
_NAMELESS = "<anon>"


def _cards(source: str | Path) -> list[SpiceCard]:
    """The cards a simulator reads of a netlist, a path or its text
    (``spice_lex.lex_deck``): not its title line, nor a card after ``.END``."""
    text = read_spice_text(source) if isinstance(source, Path) else source
    return lex_deck(text).cards


def _directive_of(card: SpiceCard) -> Directive | None:
    """The comparable form of a directive card or an ngspice ``.control`` line,
    or None for any other card and for export boilerplate."""
    if card.kind in _DIRECTIVE_KINDS:
        return parse_directive(card)
    if card.kind == "control":
        # An ngspice .control script line: not a card, but part of the deck.
        line = " ".join(card.raw_lines[0].split())
        if line and not line.startswith("*"):
            return Directive(comparable(line), line)
    return None


def read_deck(source: str | Path) -> Deck:
    """Read a netlist (a path, or its text) for a structural diff: the cards a
    simulator reads (``spice_lex.lex_deck``), so not its title line, nor a
    card after ``.END``, each in its scope.

    A definition's body is read by span (``spice_lex.subckt_children``), as
    the netlist graph reads one, so a definition nested in another is a scope
    of its own and a definition of one name in another scope is another
    scope. A name the file defines more than once in one scope is not read:
    which of them stands for which on the other side is not known (a
    simulator keeps one in force, ``spice_lex.settle_definitions``, but a
    structural diff compares files as written), so its cards are in no scope
    and ``Deck.unread`` names it.

    Raises what reading the file raises; a caller decides what an unreadable
    deck means for its comparison.
    """
    cards = _cards(source)
    openers = [(i, card) for i, card in enumerate(cards) if card.kind == "subckt"]
    # Which definitions share a scope and a name: settle_definitions' groups.
    settled = settle_definitions(
        [Definition(card.scope, card.name or _NAMELESS, "", True) for _i, card in openers],
        _AS_WRITTEN,
    )
    twice = {answer.key: answer.name for answer in settled if len(answer.members) > 1}
    scopes: dict[ScopeKey, Scope] = {(): Scope()}
    includes = False

    def read(children: list[tuple[int, SpiceCard]], key: ScopeKey, scope: Scope) -> None:
        nonlocal includes
        bounds: dict[int, str] = {}
        for i, card in children:
            if card.kind != "subckt":
                continue
            name = card.name or _NAMELESS
            folded = name.casefold()
            closer = find_matching_ends(cards, i)
            bounds.update({i: folded, **({closer: folded} if closer is not None else {})})
            if (key, folded) in twice:
                continue
            inner = Scope((*scope.names, name))
            scopes[(*key, folded)] = inner
            read(list(subckt_children(cards, i)), (*key, folded), inner)
        for i, card in children:
            if i in bounds:
                if (key, bounds[i]) not in twice and (directive := _directive_of(card)):
                    scope.definitions.setdefault(bounds[i], []).append(directive)
            elif card.kind == "instance" and card.name:
                scope.components[canon_ref(card.name)] = (card.name, instance_signature(card))
            elif (directive := _directive_of(card)) is not None:
                scope.directives.append(directive)
                if card.kind == "directive" and directive_head(card) in INCLUDE_HEADS:
                    includes = True

    read([(i, c) for i, c in enumerate(cards) if not c.scope], (), scopes[()])
    paths = {(*key[0], key[1]): name for key, name in twice.items()}
    # One inside a definition already left out is no news of its own.
    unread = tuple(
        (path, name)
        for path, name in paths.items()
        if not any(path[:depth] in paths for depth in range(1, len(path)))
    )
    return Deck(scopes, unread, includes)


def circuit_directive_delta(
    baseline: NetlistGraph,
    compared: NetlistGraph,
    calls: Iterable[tuple[FlatCall, FlatCall]],
) -> tuple[list[str], list[str]]:
    """``(added, removed)``: the directives two circuits differ by, scope by
    scope, each as :func:`directive_delta` reads one scope's.

    The scopes are the deck's top level and each subcircuit definition an
    instance both circuits have calls, wherever it is defined (inline or in an
    included file), paired by that instance as the wiring comparison paired
    them (``calls``, ``GraphComparison.calls``) whatever each side names the
    definition. A definition's directives are its body's: a ``.param`` in its
    body against one at the top level is a difference. Its ``PARAMS:``
    defaults are not among them: they reach the circuit only through the
    instances, whose parameters the wiring comparison compares as each runs
    them. Entries carry ``(in .subckt NAME)``, the name each side gives it, and
    the instance that paired them where the names differ. An instance only one
    side has, or one the wiring comparison keeps as a black box (its port
    count does not match its definition), is the wiring comparison's to
    report. An include whose file only defines subcircuits is not compared by
    its line (``NetlistGraph.definition_includes``): its definitions are
    compared themselves.
    """
    top = _scope(baseline.directives, baseline)
    added, removed = directive_delta(top, _scope(compared.directives, compared))
    # A model the baseline declares at its top level is seen inside each of its
    # definitions, so a default model one of the compared definitions declares
    # stands in for it there: a difference, not export boilerplate.
    visible = {d.model for d in top if d.model is not None}
    pairs: dict[tuple[str, str], str] = {}
    for a, b in calls:
        pairs.setdefault((a.definition, b.definition), a.ref)
    for (key_before, key_after), instance in pairs.items():
        before, after = baseline.subckts[key_before], compared.subckts[key_after]
        more, fewer = directive_delta(
            _scope(before.directives, baseline),
            _scope(after.directives, compared),
            visible=visible,
        )
        renamed = "" if before.name.casefold() == after.name.casefold() else f" of {instance}"
        added += [f"{text} (in .subckt {after.name}{renamed})" for text in more]
        removed += [f"{text} (in .subckt {before.name}{renamed})" for text in fewer]
    return list(dict.fromkeys(added)), list(dict.fromkeys(removed))


def _scope(cards: Iterable[SpiceCard], graph: NetlistGraph) -> list[Directive]:
    """One scope's directive cards, read for comparison, but for an include
    whose file only defines subcircuits (``graph.definition_includes``)."""
    bringing = {id(card) for card in graph.definition_includes}
    return [
        directive
        for card in cards
        if id(card) not in bringing and (directive := _directive_of(card)) is not None
    ]


def _by_key(directives: list[Directive]) -> dict[str, list[str]]:
    by_key: dict[str, list[str]] = {}
    for d in directives:
        by_key.setdefault(d.key, []).append(d.text)
    return by_key


def directive_delta(
    baseline: list[Directive],
    compared: list[Directive],
    *,
    visible: Collection[str] = (),
) -> tuple[list[str], list[str]]:
    """``(added, removed)``: the directives only ``compared`` has, and those only
    ``baseline`` has, as written.

    Both comparison modes read their directive difference here. A default model
    the netlister adds is left out of ``compared`` while ``baseline`` declares no
    model of that name, here or in ``visible``, the models an enclosing scope of
    the baseline declares (the module docstring says why).
    """
    declared = {d.model for d in baseline if d.model is not None} | set(visible)
    da = _by_key(baseline)
    db = _by_key([d for d in compared if not (d.default_model and d.model not in declared)])
    added = sorted(d for k in db.keys() - da.keys() for d in db[k])
    removed = sorted(d for k in da.keys() - db.keys() for d in da[k])
    return added, removed


#: What a structural diff never compares: which net each pin is on.
CONNECTIONS_NOT_COMPARED = (
    "connections: which net each pin is on is not compared; compare in "
    "equivalence mode to compare wiring"
)

#: What a structural diff does not read: the files a deck includes.
INCLUDES_NOT_COMPARED = (
    "included files: the cards of a file a deck includes (.include, .lib) are "
    "not read; only the line that includes it is compared"
)


def _in(scope: Scope) -> str:
    """The suffix an entry for an element or directive in ``scope`` carries."""
    return f" (in .subckt {scope.label})" if scope.names else ""


def structural_delta(baseline: Deck, compared: Deck) -> dict[str, list]:
    """The added/removed/changed component and directive delta, baseline →
    compared, scope by scope.

    An element or directive is its scope plus itself: the top level, or the
    definition it sits in, paired with the definition of the same name in the
    same scope on the other side (a definition only one side has is all
    added or removed). An entry inside a definition says which: a string ends
    in ``(in .subckt NAME)``, an object carries ``subckt``, nested
    definitions joined with ``/``. ``components_renamed`` lists the
    components matched only across LTspice's added ``X``; it is the one list
    here that is not a difference. ``not_compared`` says what was not
    compared: a definition either deck gives more than once in one scope, and
    included files' cards.
    """
    added: list[str] = []
    removed: list[str] = []
    renamed: list[dict[str, str]] = []
    changed: list[dict[str, str]] = []
    directives_added: list[str] = []
    directives_removed: list[str] = []
    # A definition either deck gives more than once in one scope is compared
    # on neither side: which stands for which is not known.
    unread = {key for deck in (baseline, compared) for key, _name in deck.unread}
    for key in sorted(baseline.scopes.keys() | compared.scopes.keys()):
        if any(key[:depth] in unread for depth in range(1, len(key) + 1)):
            continue
        before = baseline.scopes.get(key)
        after = compared.scopes.get(key)
        scope = after or before
        assert scope is not None
        suffix = _in(scope)
        where = {"subckt": scope.label} if scope.names else {}
        a = before.components if before else {}
        b, paired = rekey_across_added_x(a, after.components if after else {})
        added += sorted(b[ref][0] + suffix for ref in b.keys() - a.keys())
        removed += sorted(a[ref][0] + suffix for ref in a.keys() - b.keys())
        renamed += [{"before": a[ref][0], "after": b[ref][0], **where} for ref in paired]
        changed += [
            {"reference": b[ref][0], **where, "before": a[ref][1], "after": b[ref][1]}
            for ref in sorted(a.keys() & b.keys())
            if comparable(a[ref][1]) != comparable(b[ref][1])
        ]
        # A model the baseline declares in a scope around this one is seen
        # in it, so a default model declared here stands in for it.
        visible = {
            d.model
            for depth in range(len(key))
            for d in baseline.scopes.get(key[:depth], Scope()).directives
            if d.model is not None
        }
        more, fewer = directive_delta(
            _own_directives(before, key, unread),
            _own_directives(after, key, unread),
            visible=visible,
        )
        directives_added += [text + suffix for text in more]
        directives_removed += [text + suffix for text in fewer]
    sides: dict[ScopeKey, tuple[str, list[str]]] = {}
    for side, deck in (("baseline", baseline), ("compared", compared)):
        for key, name in deck.unread:
            sides.setdefault(key, ("/".join((*_names(deck, key[:-1]), name)), []))[1].append(side)
    not_compared = [
        f".subckt {label} is defined more than once in one scope in the "
        f"{' and '.join(where)} deck{'s' if len(where) > 1 else ''}, so no definition of that name there was "
        "compared on either side"
        for _key, (label, where) in sorted(sides.items())
    ]
    if baseline.includes or compared.includes:
        not_compared.append(INCLUDES_NOT_COMPARED)
    return {
        "components_added": added,
        "components_removed": removed,
        "components_renamed": renamed,
        "components_changed": changed,
        "directives_added": directives_added,
        "directives_removed": directives_removed,
        "not_compared": not_compared,
    }


def _names(deck: Deck, key: ScopeKey) -> tuple[str, ...]:
    """The names as written of the definitions scope ``key`` is inside."""
    scope = deck.scopes.get(key)
    return scope.names if scope is not None else key


def _own_directives(scope: Scope | None, key: ScopeKey, unread: set[ScopeKey]) -> list[Directive]:
    """``scope``'s directives, with the opener and closer of each definition
    in it that is compared."""
    if scope is None:
        return []
    bounds = [
        directive
        for name, directives in scope.definitions.items()
        if (*key, name) not in unread
        for directive in directives
    ]
    return [*scope.directives, *bounds]


def definitions_unread(baseline: Deck, compared: Deck) -> bool:
    """Whether either deck gives a definition more than once in one scope, so
    a structural delta between them did not compare it."""
    return bool(baseline.unread or compared.unread)


def connection_delta(
    baseline: NetlistGraph, compared: NetlistGraph
) -> tuple[list[dict[str, str | int]], list[str]]:
    """``(moved, not_compared)``: each pin the two exports of one sheet put on
    nets that do not correspond, and what about connections was not compared.

    Asks the wiring comparison the equivalence mode runs (``compare_graphs``)
    and reads its net correspondence, so a net the exporter numbered
    (``N001``, ``NC_01``, ``export_nets.named_by_export``), which another
    export may number otherwise, is matched by the parts on it and not by
    its name. A net both exports name the same (ground among them) is the
    same net: it anchors the correspondence. A pin is named only where it is
    on such a net on one side and not on it on the other, which holds
    whatever order the correspondence was built in. A move between two nets
    the exporter numbered is not named, since which pin the correspondence
    reports there depends on the order it read the parts, and
    ``not_compared`` says the wiring differs there. An entry is the part (its
    path in the flattened circuit, ``X1.M2`` inside a block), its pin by
    position on the card, counting from 1, and the net before and after, in
    lower case as the comparison reads a net name. A resistor, capacitor or
    inductor on the same named nets with its card's two nodes in the other
    order has not moved; on a numbered net, it may have, and
    ``not_compared`` says the wiring differs there. Parts only one side has are the component delta's.
    """
    named = {
        net for comp in baseline.components for net in comp.nodes if not named_by_export(net)
    } & {net for comp in compared.components for net in comp.nodes}
    comparison = compare_graphs(baseline, compared, reader=None, anchors=sorted(named))
    anchored = named | {GROUND}
    rows = [
        (ref, pin, before, after)
        for mismatch in comparison.node_partition_mismatches
        for ref, pin, before, after in mismatch.pins
    ]
    # A part whose two terminals are interchangeable, on the same nets with
    # its card's node order reversed, has moved nowhere.
    nets: dict[str, tuple[list[str], list[str]]] = {}
    for ref, _pin, before, after in rows:
        nets.setdefault(ref, ([], []))[0].append(before)
        nets[ref][1].append(after)
    turned = {
        ref
        for ref, (before, after) in nets.items()
        if canon_ref(ref).rsplit(".", 1)[-1][:1].upper() in SYMMETRIC_TWO_TERMINAL
        and sorted(before) == sorted(after)
    }
    moved = {
        (canon_ref(ref), pin): {"reference": ref, "pin": pin, "before": before, "after": after}
        for ref, pin, before, after in rows
        if before != after and (before in anchored or after in anchored) and ref not in turned
    }
    not_compared: list[str] = []
    # Only a turned part whose nets are all named is known to have moved
    # nowhere; one on a numbered net may have moved between such nets.
    certain = {ref for ref in turned if all(net in anchored for net in nets[ref][0])}
    unnamed = len(moved) < len([row for row in rows if row[0] not in certain]) or (
        not moved
        and not comparison.structurally_equivalent
        and not (comparison.added or comparison.removed or comparison.arity_errors)
    )
    if unnamed:
        not_compared.append(
            "connections: the two exports are wired otherwise among nets the exporter "
            "numbered (N001), where no pin can be named as moved; label those nets to "
            "have their pins named"
        )
    return [moved[key] for key in sorted(moved)], not_compared
