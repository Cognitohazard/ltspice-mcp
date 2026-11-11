"""Every reader of a deck's cards, held to the cards LTspice reads from it.

LTspice does not read a deck's first line, which is its title whatever it
says, nor a card after a top-level ``.end``: both builds ran
``deck/title_card``, whose title is a second ``I1``, with one ``I1``, failed
``deck/title_directive``, whose title is the ``.param`` its resistor needs,
and ran ``deck/after_end`` without the resistor after ``.end``. The first test
here holds those recordings; ``TestDeckStructure`` in
``test_recorded_ltspice_decks.py`` holds the lexer's reading
(``spice_lex.lex_deck``) to them.

Each reader below is run on those three decks and on the same decks with their
unread cards written as the cards it acts on: a title that reads as an
instance, a subcircuit call, a ``.model``, a ``.param``, an include, an
analysis, an option, a measurement, and as many cards after ``.end``. Its
answer for each must be its answer for the deck LTspice reads from it, the
title a comment and nothing after ``.end`` (``as_ltspice_reads``).

The last test is not a recording: it holds every reader to the one reading
the server takes of an included file's ``.end``, which no recording shows.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    sha256_file,
    stage_deck,
    staged_reference_targets,
)
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.experiment_inputs import capture_case_inputs, verify_case_inputs
from ltspice_mcp.lib.export_nets import exported_pin_nodes
from ltspice_mcp.lib.metrics import meas_kinds_from_netlist, noise_input_source_unit
from ltspice_mcp.lib.recovery_records import RecoveryError
from ltspice_mcp.lib.runner_base import _is_ngspice_lib_section_failure, inject_logopinfo
from ltspice_mcp.lib.services import extract_netlist_info
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.subckt_mismatch import MismatchPlanError, build_plan, closure_from_deck
from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    DeckFile,
    RandomVariation,
    VariationError,
    expand_variations,
    materialize_variants,
)
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._mismatch_fixtures import MINI_FET
from tests.test_inspect_tools import _run
from tests.test_recorded_ltspice_decks import as_ltspice_reads, deck_text
from tests.test_recovery_records import recovery_job

# --------------------------------------------------------------------------
# What LTspice read
# --------------------------------------------------------------------------

RECORDED = ["deck/title_card", "deck/title_directive", "deck/after_end"]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_ltspice_read_neither_the_title_nor_a_card_after_end(build: str, tmp_path: Path):
    """1 A into 1k is 1000 V: neither a second ``I1`` as the title nor a
    second resistor after ``.end`` was read, and a ``.param`` as the title
    left ``{r}`` undefined."""
    title_card = rec.operating_point(build, "deck/title_card", tmp_path / "title")
    assert title_card["v(a)"] == pytest.approx(1000.0)
    assert rec.entry(build, "deck/title_directive")["exit_code"] != 0
    after_end = rec.operating_point(build, "deck/after_end", tmp_path / "end")
    assert after_end["v(a)"] == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# The decks
# --------------------------------------------------------------------------

#: The cards every variant deck reads: the recordings' source into a
#: resistor, and a card of each kind a reader below acts on.
BODY = (
    "I1 0 a 1\n"
    "R1 a 0 1k\n"
    "Vg g 0 1\n"
    ".param r=1k\n"
    ".model nch nmos(vto=0.7 kp=100u)\n"
    "M1 a g 0 0 nch W=1u L=1u\n"
    ".model dx d(is=1e-14)\n"
    "D1 a 0 dx\n"
    f"{MINI_FET}"
    "XN0 a g 0 0 minifet W=1 L=0.15\n"
    ".op\n"
    ".end\n"
)

#: Cards after ``.end``, where ``deck/after_end`` has its second resistor:
#: another of each kind, a second declaration of names the deck declares, an
#: include, a library section, a second analysis, a measurement, and ``.end``
#: again.
AFTER_END = (
    "R2 a 0 1k\n"
    "M8 a g 0 0 nch W=3u L=1u\n"
    "XN8 a g 0 0 minifet W=1 L=0.15\n"
    ".model dx8 d(is=1e-14)\n"
    "D8 a 0 dx8\n"
    ".model nch nmos(vto=0.9 kp=50u)\n"
    ".param r=2k\n"
    ".include extra.inc\n"
    ".lib corners.lib tt\n"
    ".noise V(a) V9 dec 10 1 1k\n"
    ".meas op late FIND V(g)\n"
    ".end\n"
)

#: Titles, as ``deck/title_card`` and ``deck/title_directive`` have one.
TITLES = {
    "instance": "M9 a g 0 0 nch W=2u L=1u",
    "subcircuit-call": "XN9 a g 0 0 minifet W=1 L=0.15",
    "model": ".model nch nmos(vto=0.9 kp=50u)",
    "parameter": ".param r=2k",
    "include": ".include extra.inc",
    "library-section": ".lib corners.lib tt",
    "option": ".options logopinfo",
    "measurement": ".meas op late WHEN V(a)=1",
    "noise": ".noise V(a) V9 dec 10 1 1k",
    # Would end the deck before its first card, and put every card after it.
    "end": ".end",
    # Would open a subcircuit around the deck's cards, none of them then at
    # top level, and close at the definition's own .ends.
    "subcircuit": ".subckt minifet d g s b",
}

DECKS = [
    *(pytest.param(deck_text(case_id), id=case_id) for case_id in RECORDED),
    *(pytest.param(f"{title}\n{BODY}", id=f"title-{name}") for name, title in TITLES.items()),
    # The deck's only analysis is a .tran; the title would add an .op.
    pytest.param(".op\n" + BODY.replace(".op\n", ".tran 1m\n"), id="title-analysis"),
    pytest.param(f"* cards after .end\n{BODY}{AFTER_END}", id="after-end"),
]

#: Files the decks name, beside each deck: a plain include and a library with
#: a section.
SUPPORT = {
    "extra.inc": ".param extra=1\n",
    "corners.lib": ".lib tt\n.param corner=1\n.endl tt\n",
}


def write_support(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, content in SUPPORT.items():
        (where / name).write_text(content, encoding="utf-8", newline="\n")


def write_deck(where: Path, text: str) -> Path:
    write_support(where)
    deck = where / "deck.cir"
    deck.write_text(text, encoding="utf-8", newline="\n")
    return deck


def read_bodies(text: str) -> list[str]:
    """The cards LTspice reads from ``text``, as the lexer reads them."""
    return [card.body for card in lex_deck(text).cards]


# --------------------------------------------------------------------------
# The readers
# --------------------------------------------------------------------------

Reader = Callable[[str, Path, SessionState], Awaitable[Any]]
READERS: dict[str, Reader] = {}


def reader(function: Reader) -> Reader:
    READERS[function.__name__] = function
    return function


@reader
async def components(text: str, where: Path, state: SessionState) -> Any:
    """``inspect(kind="components")`` on a netlist."""
    deck = write_deck(where, text)
    (result,) = await _run(state, [{"kind": "components", "path": str(deck), "detail": "full"}])
    assert result["ok"], result
    return result["data"]["components"]


#: An assignment of each kind: a component, a ``.param`` (``r``), a model swap
#: and a per-instance mismatch delta, naming parts the decks hold where
#: LTspice reads them and where it does not.
ASSIGNMENTS = {
    "I1": "2",
    "R1": "2k",
    "R2": "2k",
    "M8": "2k",
    "M9": "2k",
    "r": "2k",
    "D*@model": "dx2",
    "XN0:delvto": 0.01,
    "XN8:delvto": 0.01,
    "XN9:delvto": 0.01,
}


def materialized(text: str, where: Path, variation: AssignVariation | RandomVariation) -> Any:
    """The case one variation writes, as LTspice reads it, with its receipt;
    or the code it is refused with. The deck is staged, and the variation
    reaches the files it includes, as ``run_experiments`` hands them over."""
    deck = write_deck(where, text)
    try:
        staged = stage_deck(deck, where / "stage", [where], origin=deck)
    except DeckStagingError as exc:
        return exc.code
    circuit = CircuitDeck(
        "dut",
        staged.staged_deck,
        staged.text,
        includes=tuple(
            DeckFile(file.staged_path, file.text, file.sha256, file.codec)
            for file in staged.includes
        ),
        codec=staged.codec,
    )
    try:
        (case,) = materialize_variants(
            circuit, expand_variations([circuit], [variation]), where / "case"
        )
    except VariationError as exc:
        return exc.code
    # A case names the files it includes by their staged paths, under ``where``.
    return case.assignments, [
        body.replace(str(where), "WHERE").replace(where.as_posix(), "WHERE")
        for body in read_bodies(case.text)
    ]


@reader
async def assignments(text: str, where: Path, state: SessionState) -> Any:
    """``run_experiments``' assign variations."""
    return {
        target: materialized(
            text,
            where / str(index),
            AssignVariation(kind="assign", assign={target: [value]}),
        )
        for index, (target, value) in enumerate(ASSIGNMENTS.items())
    }


#: A random rule of each kind: components, a ``.param``, a model parameter,
#: mismatch on top-level transistors, and mismatch reached through a
#: subcircuit.
RANDOM_RULES: dict[str, dict[str, Any]] = {
    "component": {"rule": "component", "target": "R*", "tolerance": 0.1},
    "param": {"rule": "param", "target": "r", "tolerance": 0.1},
    "model": {"rule": "model", "target": "nch", "param": "VTO", "tolerance": 0.1},
    "mismatch": {"rule": "mismatch", "prefix": "M", "AVT": 3e-3, "AK": 0.02},
    "subcircuit-mismatch": {"rule": "mismatch", "prefix": "XN", "AVT": 3e-3},
}


@reader
async def random_rules(text: str, where: Path, state: SessionState) -> Any:
    """``run_experiments``' random variations."""
    return {
        name: materialized(
            text,
            where / name,
            RandomVariation.model_validate(
                {"kind": "random", "runs": 1, "seed": 1, "rules": [rule]}
            ),
        )
        for name, rule in RANDOM_RULES.items()
    }


@reader
async def staging(text: str, where: Path, state: SessionState) -> Any:
    """The files a run's snapshot takes in."""
    deck = write_deck(where, text)
    try:
        staged = stage_deck(deck, where / "stage", [where], origin=deck)
    except DeckStagingError as exc:
        return exc.code
    return sorted((entry.path.name, entry.section or "") for entry in staged.manifest)


@reader
async def staged_references(text: str, where: Path, state: SessionState) -> Any:
    """The files a staged deck names, which a case's copies are routed by."""
    deck = write_deck(where, text)
    return sorted(path.name for path in staged_reference_targets(text, deck, depth=0))


@reader
async def mismatch_closure(text: str, where: Path, state: SessionState) -> Any:
    """The files the per-instance mismatch plan reads for one deck."""
    deck = write_deck(where, text)
    return [file.path.name for file in closure_from_deck(deck, text)]


@reader
async def mismatch_plan(text: str, where: Path, state: SessionState) -> Any:
    """The devices a mismatch prefix reaches through a subcircuit."""
    deck = write_deck(where, text)
    try:
        plan = build_plan(closure_from_deck(deck, text), prefix="XN")
    except MismatchPlanError as exc:
        return exc.code
    return [target.ref for target in plan.targets]


@reader
async def operating_point_option(text: str, where: Path, state: SessionState) -> Any:
    """The deck an LTspice ``.op`` run is handed, with ``.options logopinfo``
    added when it asks for an operating point."""
    deck = write_deck(where, text)
    run = inject_logopinfo(deck, rec.launcher("ltspice26"), "job")
    return read_bodies(read_spice_text(run))


@reader
async def measurement_kinds(text: str, where: Path, state: SessionState) -> Any:
    return meas_kinds_from_netlist(write_deck(where, text))


@reader
async def noise_unit(text: str, where: Path, state: SessionState) -> Any:
    return noise_input_source_unit(write_deck(where, text))


@reader
async def ngspice_library_section(text: str, where: Path, state: SessionState) -> Any:
    """Whether a failed ngspice run is put down to a sectioned ``.lib``."""
    return _is_ngspice_lib_section_failure(write_deck(where, text), NGspiceSimulator, "kiltpsa")


#: Parts as a sheet names them, with their pins in SpiceOrder.
EXPORTED_PARTS = [
    ("I1", "I", ("P", "N")),
    ("R1", "R", ("A", "B")),
    ("R2", "R", ("A", "B")),
    ("M8", "M", ("D", "G", "S", "B")),
    ("M9", "M", ("D", "G", "S", "B")),
]


@reader
async def exported_pins(text: str, where: Path, state: SessionState) -> Any:
    """The node an export puts each pin of a sheet's part on."""
    return exported_pin_nodes(text, EXPORTED_PARTS)


@reader
async def netlist_info(text: str, where: Path, state: SessionState) -> Any:
    return extract_netlist_info(write_deck(where, text))["components"]


@reader
async def recovery(text: str, where: Path, state: SessionState) -> Any:
    """Whether a seeded case's frozen inputs can be recovered: its staged
    deck's includes each frozen, and one analysis."""
    job = recovery_job(where)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    write_support(job.output_folder)
    case.staged_deck.write_text(text, encoding="utf-8", newline="\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    try:
        captured = capture_case_inputs(
            case, job.sources[0], lineage_root=job.output_folder, seeded=True
        )
        verify_case_inputs(captured, seeded=True)
    except RecoveryError as exc:
        return exc.code
    return "recoverable"


# --------------------------------------------------------------------------
# The test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", DECKS)
@pytest.mark.parametrize("name", sorted(READERS))
async def test_each_reader_reads_the_deck_ltspice_reads(
    name: str, text: str, tmp_path: Path, state_no_sim: SessionState
):
    read = as_ltspice_reads(text)
    assert read != text
    expected = await READERS[name](read, tmp_path / "read", state_no_sim)
    assert await READERS[name](text, tmp_path / "deck", state_no_sim) == expected


# --------------------------------------------------------------------------
# An included file's .end
# --------------------------------------------------------------------------

#: A deck that includes ``part.inc`` ahead of the cards above.
INCLUDING = f"* a deck with an include\n.include part.inc\n{BODY}"

#: The included file up to its ``.end``.
INCLUDED = "R5 a 0 1k\n.end\n"


@pytest.mark.parametrize("name", sorted(READERS))
async def test_each_reader_reads_an_included_file_up_to_its_end(
    name: str, tmp_path: Path, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    """Not a recording: no recorded input shows what LTspice does with a card
    after an included file's ``.end``. The hierarchy, the run's requirements
    and the netlist graph leave such a card out (``lex_deck(title=False)``),
    and every reader takes that same reading, so none acts on a card another
    does not see. An assignment or a random rule named ``R2``, ``r`` or
    ``nch`` after the included ``.end`` and reached it, and staging and the
    mismatch plan followed its include, while ``lex_deck`` read only ``R5``.
    Each reader's answer is its answer for the same file with nothing after
    its ``.end``."""
    included = INCLUDED + AFTER_END
    assert [card.body for card in lex_deck(included, title=False).cards] == [
        card.body for card in lex_deck(INCLUDED, title=False).cards
    ]
    monkeypatch.setitem(SUPPORT, "part.inc", INCLUDED)
    expected = await READERS[name](INCLUDING, tmp_path / "read", state_no_sim)
    monkeypatch.setitem(SUPPORT, "part.inc", included)
    assert await READERS[name](INCLUDING, tmp_path / "file", state_no_sim) == expected
