"""What the server reads a deck to say, on every route that reads its cards.

Shared by the grids of what a deck says (``test_deck_card_grid.py``,
``test_element_nodes_grid.py``, ``test_declaration_grid.py``). Each route is
the server's own code reading a deck file as the tool or module that uses it
reads one; nothing is substituted for it. A route answers, for each thing a
cell asks about (an element, a parameter, a model's parameter or level, a
subcircuit's default, a directive), what it read, or nothing where it does not
read that kind of thing or declines the deck.

A cell names its subject as words: ``element R1``, ``param pa``,
``model swp ron``, ``level mcomp``, ``default simp r``, ``subckt cgsub``,
``meas m08``. An answer is ``"absent"`` (the route reads the deck and holds no
such thing), a number (the value it reads it to take), ``"present"`` (it holds
the thing and reads no number for it), or None (the route does not answer).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.deck_staging import scan_include_references
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import SAME_VALUE_RTOL, plain_number
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.netlist_graph import parse_netlist_graph
from ltspice_mcp.lib.result_observations import parse_source_amplitudes
from ltspice_mcp.lib.services import extract_netlist_info
from ltspice_mcp.lib.spice_lex import (
    INCLUDE_HEADS,
    SpiceCard,
    SpiceLexError,
    directive_head,
    lex_deck,
    lex_for_edit,
)
from ltspice_mcp.lib.spice_lex_views import (
    MeasCard,
    ModelCard,
    ParamCard,
    SubcktCard,
    assigned_params,
    read_instance,
)
from ltspice_mcp.lib.spice_validator import _instance_terminals
from ltspice_mcp.lib.variations import _deck_targets
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, dispatched

#: The names a reader gives ground, left out of a card's nodes when they are
#: compared with the node voltages a results file holds, which has none.
GROUND = frozenset({"0", "gnd"})

Answer = str | float | None


@dataclass(frozen=True)
class Element:
    """One element card as a route reads it. ``nodes`` is None where the
    route reads no nodes; ``number`` is the value it reads the element to take,
    None where it reads none."""

    nodes: tuple[str, ...] | None = None
    model: str | None = None
    number: float | None = None
    has_number: bool = True
    #: False where the route says it cannot read the element's model.
    model_read: bool = True


@dataclass
class Reading:
    """What one route read of a deck. A field left None is a kind of thing
    the route does not read; ``letters`` limits the elements it answers for
    to those letters; ``declined`` says why it refused the deck."""

    route: str
    elements: dict[str, Element] | None = None
    letters: str | None = None
    params: dict[str, float | None] | None = None
    models: dict[str, dict[str, str]] | None = None
    levels: dict[str, float | None] | None = None
    defaults: dict[str, dict[str, str]] | None = None
    subckts: set[str] | None = None
    meas: set[str] | None = None
    directives: dict[str, str] | None = None
    declined: str | None = None

    def answer(self, subject: str) -> Answer:
        """What the route reads ``subject`` to be."""
        if self.declined is not None:
            return None
        kind, *words = subject.lower().split()
        if kind == "element":
            (ref,) = words
            if self.elements is None or (
                self.letters is not None and ref[:1].upper() not in self.letters
            ):
                return None
            element = self.elements.get(ref)
            if element is None:
                return "absent"
            if not element.has_number:
                return None
            return element.number if element.number is not None else "present"
        if kind == "param":
            (name,) = words
            if self.params is None:
                return None
            if name not in self.params:
                return "absent"
            value = self.params[name]
            return "present" if value is None else value
        if kind == "model":
            name, key = words
            if self.models is None:
                return None
            if key not in self.models.get(name, {}):
                return "absent"
            number = plain_number(self.models[name][key])
            return "present" if number is None else number
        if kind == "level":
            (name,) = words
            if self.levels is None or name not in self.levels:
                return None if self.levels is None else "absent"
            level = self.levels[name]
            return "present" if level is None else level
        if kind == "default":
            name, key = words
            if self.defaults is None:
                return None
            if key not in self.defaults.get(name, {}):
                return "absent"
            number = plain_number(self.defaults[name][key])
            return "present" if number is None else number
        if kind == "subckt":
            (name,) = words
            return (
                None if self.subckts is None else "present" if name in self.subckts else "absent"
            )
        if kind == "meas":
            (name,) = words
            return None if self.meas is None else "present" if name in self.meas else "absent"
        raise AssertionError(f"no subject {subject!r}")

    def directive(self, written: str) -> str | None:
        """Which directive the route reads the card that starts with
        ``written`` (as the deck writes it) to be, None where it reads none."""
        if self.directives is None or self.declined is not None:
            return None
        return self.directives.get(written, "absent")


def directive_of(card: SpiceCard) -> str:
    """Which directive the lexer reads ``card`` as: its kind where the lexer
    gives directives a kind of their own (``param``, ``model``, ``subckt``,
    ``ends``, ``end``, ``meas``, ``control``), ``include`` or ``lib`` for an
    include head of ``INCLUDE_HEADS``, else its head without the dot."""
    if card.kind in ("param", "model", "subckt", "ends", "end", "meas"):
        return card.kind
    if card.kind == "control":
        return "control"
    head = directive_head(card)
    if head in INCLUDE_HEADS:
        return "lib" if head in (".lib", ".libfile") else "include"
    return head.removeprefix(".")


def _first_word(card: SpiceCard) -> str:
    text = card.raw_lines[0].strip() if card.kind == "control" else card.body.strip()
    return text.split(None, 1)[0] if text else ""


def _instances(cards: Iterable[SpiceCard]) -> dict[str, Element]:
    found: dict[str, Element] = {}
    for card in cards:
        if card.kind != "instance" or card.scope or not card.name:
            continue
        line = read_instance(card)
        if line is None:
            found[card.name.lower()] = Element(nodes=None, number=None)
            continue
        number = plain_number(line.value) if line.value is not None else None
        found[line.ref.lower()] = Element(
            nodes=tuple(n.lower() for n in line.nodes),
            model=line.model.lower() if line.model else None,
            number=number,
        )
    return found


def _declarations(cards: list[SpiceCard], route: str) -> Reading:
    """Everything the lexer and its views read of ``cards``."""
    params: dict[str, float | None] = {}
    models: dict[str, dict[str, str]] = {}
    levels: dict[str, float | None] = {}
    defaults: dict[str, dict[str, str]] = {}
    subckts: set[str] = set()
    meas: set[str] = set()
    directives: dict[str, str] = {}
    for card in cards:
        if card.kind not in ("instance", "comment", "blank"):
            directives[_first_word(card)] = directive_of(card)
        try:
            if card.kind == "param":
                for name, token in assigned_params(card):
                    params[name.lower()] = plain_number(token.value or "")
            elif card.kind == "model":
                view = ModelCard.from_card(card)
                models[view.name.lower()] = {k.lower(): v for k, v in view.params.items()}
                levels[view.name.lower()] = view.level
            elif card.kind == "subckt":
                sub = SubcktCard.from_card(card)
                subckts.add(sub.name.lower())
                defaults[sub.name.lower()] = {k.lower(): v for k, v in sub.param_defaults.items()}
            elif card.kind == "meas":
                meas.add(MeasCard.from_card(card).name.lower())
        except SpiceLexError:
            continue
    return Reading(
        route,
        elements=_instances(cards),
        params=params,
        models=models,
        levels=levels,
        defaults=defaults,
        subckts=subckts,
        meas=meas,
        directives=directives,
    )


def _guarded(route: str, read: Callable[[], Reading]) -> Reading:
    try:
        return read()
    except Exception as exc:  # a route that refuses the deck answers nothing
        return Reading(route, declined=f"{type(exc).__name__}: {exc}")


def _lexed(deck: Path) -> Reading:
    return _declarations(lex_deck(read_spice_text(deck)).cards, "lex_deck")


def _edited(deck: Path) -> Reading:
    return _declarations(lex_for_edit(read_spice_text(deck), deck=True).read, "lex_for_edit")


def _with_includes(deck: Path, family: str) -> Reading:
    """The deck's cards and those of every file it includes, each file lexed
    as the server lexes an included file (``lex_deck`` with no title), the
    files found by ``scan_include_references``."""
    cards = lex_deck(read_spice_text(deck)).cards
    every = list(cards)
    for reference in scan_include_references(cards, deck, family=family):
        path = deck.parent / reference.raw_path.strip("\"'")
        if path.is_file():
            every += lex_deck(read_spice_text(path), title=False).cards
    reading = _declarations(every, "lex_deck with its included files")
    return reading


def _param_cards(deck: Path) -> Reading:
    params: dict[str, float | None] = {}
    for card in lex_deck(read_spice_text(deck)).cards:
        if card.kind == "param":
            try:
                view = ParamCard.from_card(card)
            except SpiceLexError:
                continue
            params[view.name.lower()] = plain_number(view.value)
    return Reading("ParamCard", params=params)


def _netlist_info(deck: Path) -> Reading:
    rows = extract_netlist_info(deck)["components"]
    return Reading(
        "extract_netlist_info",
        elements={
            row["reference"].lower(): Element(number=plain_number(row["value"])) for row in rows
        },
    )


def _graph(deck: Path, profile: SemanticProfile) -> Reading:
    graph = parse_netlist_graph(deck, reader=profile)
    return Reading(
        "parse_netlist_graph",
        elements={
            c.ref.lower(): Element(
                nodes=tuple(n.lower() for n in c.nodes),
                model=c.model.lower() if c.model else None,
                number=plain_number(c.value) if c.value is not None else None,
            )
            for c in graph.components
        },
        subckts={name.lower() for name in graph.subckts},
    )


def _hierarchy(deck: Path, profile: SemanticProfile) -> Reading:
    found = load_hierarchy(str(deck), None, profile)
    elements: dict[str, Element] = {}
    levels: dict[str, float | None] = {}
    for row in found.instances:
        if len(row.instance) != 1:
            continue
        # A row whose nodes or model the hierarchy says it cannot read
        # (``connectivity_reason``, ``model_reason``) answers nothing of them.
        elements[row.reference.lower()] = Element(
            nodes=None if row.connectivity_reason else tuple(n.name.lower() for n in row.nodes),
            model=row.model_name.lower() if row.model_name else None,
            number=row.value.value,
            model_read=row.model_reason is None,
        )
        for model in row.model_family:
            levels[model.name.lower()] = model.level
    return Reading("load_hierarchy", elements=elements, levels=levels)


def _terminals(deck: Path) -> Reading:
    """The nodes the dangling-node check counts: ``_instance_terminals``,
    over the cards ``lex_deck`` reads."""
    elements: dict[str, Element] = {}
    for card in lex_deck(read_spice_text(deck)).cards:
        if card.kind != "instance" or card.scope or not card.name:
            continue
        line = read_instance(card)
        if line is not None:
            terminals, _ = _instance_terminals(line)
            elements[line.ref.lower()] = Element(
                nodes=tuple(t.lower() for t in terminals), has_number=False
            )
    return Reading("spice_validator._instance_terminals", elements=elements)


def _amplitudes(deck: Path) -> Reading:
    """The peak of each voltage source, which a run's trigger compares node
    voltages against; it reads ``.param`` constants on its own."""
    found = parse_source_amplitudes(read_spice_text(deck))
    if not found:
        return Reading("parse_source_amplitudes", declined="no source it can bound")
    return Reading(
        "parse_source_amplitudes",
        elements={ref.lower(): Element(number=value) for ref, value in found.items()},
        letters="V",
    )


def _targets(deck: Path) -> Reading:
    """What a variation can target: the names of the deck's parameters and
    components (``variations._deck_targets``)."""
    targets = _deck_targets(read_spice_text(deck), deck, 0)
    return Reading(
        "variation targets",
        elements={name.lower(): Element(has_number=False) for name in targets.components},
        params={name.lower(): None for name in targets.params},
    )


async def _inspected(state: SessionState, deck: Path) -> list[Reading]:
    query = {"kind": "components", "path": str(deck), "detail": "full"}
    readings = []
    for route, call in (
        ("inspect components", lambda: dispatched(state, "inspect", queries=[query])),
        (
            "api inspect components",
            lambda: asyncio.to_thread(lambda: SyncApi(state).inspect(queries=[query])),
        ),
    ):
        reply: dict[str, Any] = await call()
        (result,) = reply["results"]
        if not result["ok"]:
            readings.append(Reading(route, declined=result["error"]["message"]))
            continue
        elements = {}
        for row in result["data"]["components"]:
            number = plain_number(row["value"]) if row.get("value") else None
            elements[row["reference"].lower()] = Element(
                nodes=tuple(n.lower() for n in row["nodes"]) if "nodes" in row else None,
                model=row["model"].lower() if row.get("model") else None,
                number=number,
            )
        readings.append(Reading(route, elements=elements))
    return readings


@dataclass
class DeckReadings:
    """Every route's reading of one deck."""

    readings: list[Reading] = field(default_factory=list)

    def answers(self, subject: str) -> dict[str, Answer]:
        """Each route that answers ``subject``, with its answer."""
        found = {r.route: r.answer(subject) for r in self.readings}
        return {route: answer for route, answer in found.items() if answer is not None}

    def directives(self, written: str) -> dict[str, str]:
        found = {r.route: r.directive(written) for r in self.readings}
        return {route: d for route, d in found.items() if d is not None}

    def nodes(self, ref: str) -> dict[str, tuple[str, ...]]:
        """Each route's nodes of element ``ref``, ground left out."""
        found = {}
        for r in self.readings:
            if r.declined is not None or r.elements is None:
                continue
            element = r.elements.get(ref.lower())
            if element is not None and element.nodes is not None:
                found[r.route] = tuple(n for n in element.nodes if n not in GROUND)
        return found

    def models(self, ref: str) -> dict[str, str | None]:
        found = {}
        for r in self.readings:
            if r.declined is not None or r.elements is None:
                continue
            element = r.elements.get(ref.lower())
            if element is not None and element.nodes is not None and element.model_read:
                found[r.route] = element.model
        return found


def read_offline(
    deck: Path, *, simulator: str = "ltspice", mode: str | None = None
) -> DeckReadings:
    """Every route that needs no session, for the reader of ``simulator``."""
    profile = SemanticProfile(simulator, mode)  # type: ignore[arg-type]
    return DeckReadings(
        [
            _guarded("lex_deck", lambda: _lexed(deck)),
            _guarded("lex_for_edit", lambda: _edited(deck)),
            _guarded("lex_deck with its included files", lambda: _with_includes(deck, simulator)),
            _guarded("ParamCard", lambda: _param_cards(deck)),
            _guarded("extract_netlist_info", lambda: _netlist_info(deck)),
            _guarded("parse_netlist_graph", lambda: _graph(deck, profile)),
            _guarded("load_hierarchy", lambda: _hierarchy(deck, profile)),
            _guarded("spice_validator._instance_terminals", lambda: _terminals(deck)),
            _guarded("parse_source_amplitudes", lambda: _amplitudes(deck)),
            _guarded("variation targets", lambda: _targets(deck)),
        ]
    )


async def read_every_route(state: SessionState, deck: Path) -> DeckReadings:
    """Every route, the tool's and the Python API's among them, for LTspice."""
    readings = await asyncio.to_thread(read_offline, deck)
    readings.readings += await _inspected(state, deck)
    return readings


def same_reading(a: Answer, b: Answer) -> bool:
    """Whether two answers are one reading: the same word, or numbers equal
    within what a results file stores, or ``present`` beside a number."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=SAME_VALUE_RTOL, abs_tol=1e-12)
    if "present" in (a, b):
        other = b if a == "present" else a
        return other not in ("absent", "refused")
    return a == b


# What LTspice and ngspice did --------------------------------------------------

#: Routes that read a card in a file the deck includes; every other route
#: reads the deck's own file, so it does not answer for such a card.
FOLLOWS_INCLUDES = frozenset({"lex_deck with its included files", "load_hierarchy"})

#: How close a voltage must be to the one a reading gives: a raw stores a
#: node voltage in four bytes, and a cell's volts are written to six digits.
VOLTS_RTOL = 1e-5


def stage_case(case_id: str, directory: Path) -> Path:
    """Copy a case's deck into ``directory`` with the files copied beside it
    for the recording, and return the deck."""
    from tests import _ltspice_recorded as rec
    from tests.ltspice_recorder import INPUTS

    case = rec.CASES.case(case_id)
    directory.mkdir(parents=True, exist_ok=True)
    deck = directory / Path(case.source).name
    deck.write_bytes((INPUTS / case.source).read_bytes())
    for extra in case.extra:
        beside = directory / case.copies[extra]
        beside.parent.mkdir(parents=True, exist_ok=True)
        beside.write_bytes((INPUTS / extra).read_bytes())
    return deck


def recorded_voltages(build: str, case_id: str, scratch: Path) -> dict[str, float] | None:
    """Every node voltage of a recorded run at its first point, by
    lower-cased trace name, the real part of an AC point; None where the
    build refused the deck."""
    from tests import _ltspice_recorded as rec

    entry = rec.entry(build, case_id)
    if entry["exit_code"] != 0 or not rec.has(build, f"{case_id}.raw"):
        return None
    raw = rec.decode(build, case_id, scratch).raw
    return {name.lower(): complex(raw.get_wave(name, 0)[0]).real for name in raw.get_trace_names()}


def recorded_printed(build: str, case_id: str) -> set[str]:
    """The measurements a recorded run's log printed, by name in lower case."""
    from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
    from tests import _ltspice_recorded as rec

    if not rec.has(build, f"{case_id}.log"):
        return set()
    return set(printed_measurements(read_log_text(rec.recorded(build, f"{case_id}.log"))))


def _matches(spec: Any, node: str, voltages: dict[str, float] | None) -> bool:
    if spec == "refused":
        return voltages is None
    if voltages is None:
        return False
    v = voltages.get(f"v({node.lower()})")
    if spec == "missing":
        return v is None
    if v is None:
        return False
    if spec == "present":
        return True
    if isinstance(spec, str):
        other = voltages.get(f"v({spec.lstrip('!=').lower()})")
        if other is None:
            return False
        same = math.isclose(v, other, rel_tol=VOLTS_RTOL, abs_tol=1e-9)
        return not same if spec.startswith("!=") else same
    return math.isclose(v, float(spec), rel_tol=VOLTS_RTOL, abs_tol=1e-9)


def reading_of(volts: dict[str, Any], node: str, voltages: dict[str, float] | None) -> str | None:
    """Which of a cell's readings ``voltages`` show (the key of ``volts``
    whose voltage the node has), None when they show none of them."""
    for key, spec in volts.items():
        for one in spec if isinstance(spec, list) else [spec]:
            if _matches(one, node, voltages):
                return key
    return None


def as_answer(key: str) -> Answer:
    """A reading's key as an answer: a word, or the number it names."""
    try:
        return float(key)
    except ValueError:
        return key


def declared_reading(cell: Any, build: str) -> str | None:
    """The reading a cell's form means on ``build``, None where it declares none."""
    reads = cell.extra.get("reads")
    return reads.get(build) if isinstance(reads, dict) else reads


def differs_declared(cell: Any, build: str) -> set[tuple[str, str]]:
    return {
        (cell.item, route)
        for entry in cell.extra.get("differs", ())
        if entry["build"] == build
        for route in entry["routes"]
    }


def ngspice_run(deck: Path, mode: str) -> tuple[dict[str, float] | None, set[str]]:
    """Run ``deck`` through the ngspice on PATH in ``mode`` from the deck's
    folder, as the server asks ngspice (``ngspice_mode.run_batch``): its node
    voltages at the operating point it prints, None where it printed none,
    and the measurements it printed."""
    import re

    from ltspice_mcp.lib.ngspice_mode import run_batch
    from tests.conftest import LIVENESS_S

    log = run_batch(
        ("ngspice",),
        mode,
        deck.read_bytes(),
        cwd=deck.parent,
        clean_startup=True,
        timeout=LIVENESS_S,
    )
    table = re.search(
        r"Node\s+Voltage\s*\n\s*-+\s+-+\s*\n(?:\s*-+\s+-+\s*\n)?(.*?)\n\s*\n", log, re.S
    )
    printed = {m.lower() for m in re.findall(r"^(\w+)\s*=\s*\S+", log, re.M)}
    if table is None:
        return None, printed
    voltages = {}
    for line in table.group(1).splitlines():
        words = line.split()
        if len(words) == 2:
            try:
                voltages[f"v({words[0].lower()})"] = float(words[1])
            except ValueError:
                continue
    return voltages, printed


#: The compatibility modes the server models (``lib/ngspice_compat.py``).
NGSPICE_MODES = ("", "hsa", "kiltpsa", "lt", "ps", "spe")
