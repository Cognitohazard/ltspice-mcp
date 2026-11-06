"""How the server reads a deck, held against what LTspice did with the same deck.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: decks
LTspice 26 and LTspice XVII each ran. Most are operating points of a 1 A
current source into one resistor, so the node voltage in the recorded raw is
the resistance LTspice read off the card. The server's value parser, lexer,
lint and arity rules read the same card and have to reach the same answer.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib import deck_prep, montecarlo
from ltspice_mcp.lib.encoding import (
    decode_spice_bytes_with_encoding,
    read_spice_text,
    read_spice_text_with_encoding,
    rewrite_codec,
)
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.hierarchy_expr import evaluate
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.log_types import MeasurementsOutput
from ltspice_mcp.lib.netlist_diff import read_deck, structural_delta
from ltspice_mcp.lib.netlist_graph import (
    compare_graphs,
    flatten_graph,
    parse_netlist_graph,
    values_equal,
)
from ltspice_mcp.lib.result_observations import parse_requested_outputs, parse_source_amplitudes
from ltspice_mcp.lib.runner_base import deck_requirements
from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build, is_cp1252_ltspice_executable
from ltspice_mcp.lib.spice_lex import emit, lex, lex_deck, lex_for_edit
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import ModelCard, read_instance
from ltspice_mcp.lib.spice_validator import (
    _instance_terminals,
    validate_directive,
    validate_netlist_arity,
    validate_netlist_bias_topology,
    validate_netlist_dangling_nodes,
)
from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    RandomVariation,
    VariationError,
    expand_variations,
    materialize_variants,
)
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS
from tests.test_inspect_tools import _run

MICRO = "µ"


def deck_text(case_id: str) -> str:
    return read_spice_text(INPUTS / rec.CASES.case(case_id).source)


def lint_input(case_id: str, **options) -> list[dict]:
    """The linter's findings on a recorded input, read in the codec staging
    would write it in, as ``run_experiments`` lints it."""
    path = INPUTS / rec.CASES.case(case_id).source
    text, encoding = read_spice_text_with_encoding(path)
    return lint_deck(text, path, "ltspice", "LTspice", codec=rewrite_codec(encoding), **options)


def resistances(case_id: str) -> dict[str, str]:
    """Each probe node of a suffix deck and the value written on its resistor."""
    values: dict[str, str] = {}
    for card in lex_deck(deck_text(case_id)).cards:
        line = read_instance(card) if card.kind == "instance" else None
        if line is not None and line.ref[:1].upper() == "R":
            values[line.nodes[0].lower()] = line.value or ""
    return values


def ran(build: str, case_id: str) -> bool:
    return rec.entry(build, case_id)["exit_code"] == 0


# --------------------------------------------------------------------------
# Value suffixes
# --------------------------------------------------------------------------

SUFFIX_DECKS = [
    "deck/suffixes",
    "deck/suffix_unknown",
    "deck/suffix_shorthand",
    "deck/suffix_prefix_match",
]

#: Spellings the server's parser refused until it read values as LTspice
#: does: a letter that is no scale suffix, which LTspice skips with what
#: follows it, and the "3k4" shorthand that puts the suffix where the decimal
#: point goes. What LTspice made of each is the recording's to say.
ONCE_REFUSED = frozenset(
    {"2Hz", "3V", "2ohm", "2A", "7x", "5s", "2Ohms", "4H", "6W", "9V1"}
    | {"1k5", "4R7", "2M2", "3u3", "1Meg5"}
)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
def test_a_value_the_server_parses_is_the_value_ltspice_read(
    build: str, case_id: str, tmp_path: Path
):
    """Scale suffixes in either case, ``M`` against ``Meg``, ``mil``, unit
    tails, the ``1k5`` shorthand, and words that merely begin with a suffix
    (``1MHz`` is a millihertz to LTspice, ``1milli`` a mil)."""
    voltages = rec.operating_point(build, case_id, tmp_path)
    written = resistances(case_id)
    assert len(written) >= 5
    for node, spelling in written.items():
        # A raw stores a node voltage in four bytes.
        assert parse_spice_value(spelling) == pytest.approx(voltages[f"v({node})"], rel=1e-6), (
            spelling
        )


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
def test_every_reader_of_a_value_that_runs_reads_the_value_ltspice_read(
    build: str, case_id: str, tmp_path: Path
):
    """The equivalence comparison and the Monte Carlo nominal read a value as
    LTspice runs it, so ``4R7`` equals 4.7 and a random rule on it draws
    around 4.7."""
    voltages = rec.operating_point(build, case_id, tmp_path)
    for node, spelling in resistances(case_id).items():
        ran_as = voltages[f"v({node})"]
        assert montecarlo.parse_value(spelling) == pytest.approx(ran_as, rel=1e-6), spelling
        assert values_equal(spelling, repr(ran_as), rtol=1e-6), spelling


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
def test_a_deck_is_the_same_circuit_with_its_values_written_as_ltspice_ran_them(
    build: str, case_id: str, tmp_path: Path
):
    voltages = rec.operating_point(build, case_id, tmp_path)
    cards = lex(deck_text(case_id)).cards
    for line in map(read_instance, cards):
        if line is not None and line.ref[:1] == "R":
            line.set_value(repr(voltages[f"v({line.nodes[0].lower()})"]))
    comparison = compare_graphs(deck_text(case_id), emit(cards))
    assert comparison.equivalent, comparison.value_mismatches


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
def test_a_random_rule_draws_around_the_value_ltspice_ran(
    build: str, case_id: str, tmp_path: Path
):
    voltages = rec.operating_point(build, case_id, tmp_path / "raw")
    path = tmp_path / "deck.cir"
    path.write_text(deck_text(case_id), encoding="utf-8")
    circuit = CircuitDeck("dut", path, deck_text(case_id))
    rule = {"rule": "component", "target": "R*", "tolerance": 1e-9, "distribution": "uniform"}
    variation = RandomVariation.model_validate(
        {"kind": "random", "runs": 1, "seed": 1, "rules": [rule]}
    )
    (case,) = materialize_variants(
        circuit, expand_variations([circuit], [variation]), tmp_path / "out"
    )
    drawn = {
        name.removeprefix("random:component:"): value
        for name, value in case.assignments.items()
        if name.startswith("random:component:")
    }
    assert drawn == {
        line.ref: pytest.approx(voltages[f"v({line.nodes[0].lower()})"], rel=1e-6)
        for line in map(read_instance, lex_deck(deck_text(case_id)).cards)
        if line is not None and line.ref[:1] == "R"
    }


def test_the_spellings_once_refused_are_all_recorded():
    """Each spelling above is in a recorded deck, so the test before this one
    holds the server to LTspice's reading of it."""
    spellings = {
        spelling
        for case_id in SUFFIX_DECKS
        for spelling in resistances(case_id).values()
        if spelling in ONCE_REFUSED
    }
    assert spellings == ONCE_REFUSED


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
async def test_a_value_the_hierarchy_resolves_is_the_value_ltspice_read(
    build: str, case_id: str, tmp_path: Path, state_no_sim, work_dir: Path
):
    """``inspect(kind="hierarchy")`` reads a resistor's value through the
    expression evaluator, which once took ``4R7`` for 4 followed by a stray 7
    and answered 4.07."""
    voltages = rec.operating_point(build, case_id, tmp_path)
    rows = await hierarchy_rows(state_no_sim, work_dir, case_id)
    resistors = {path: row for path, row in rows.items() if path[0][:1].upper() == "R"}
    assert len(resistors) == len(resistances(case_id))
    for (reference,), row in resistors.items():
        node = row["nodes"][0]["name"].lower()
        assert row["value"]["value"] == pytest.approx(voltages[f"v({node})"], rel=1e-6), (
            reference,
            row["value"]["expression"],
        )


def test_a_digit_after_the_letters_is_declined_inside_an_expression():
    """In braces LTspice's reading of ``1k5`` is not recorded, so the evaluator
    claims no value for it rather than read ``1k`` and a stray 5. The digits of
    an exponent are not after the letters."""

    def ltspice(text: str, **options) -> float:
        return evaluate(text, lambda name: 0.0, simulator="ltspice", **options)

    with pytest.raises(ValueError, match="1k5"):
        ltspice("{1k5}")
    assert ltspice("{2*1k}") == pytest.approx(2000.0)
    assert ltspice("{1e5}") == pytest.approx(1e5)
    assert ltspice("{2.5E3k*2}") == pytest.approx(5e6)
    # A bare element value is held to the same bounds as an expression.
    with pytest.raises(ValueError, match="finite"):
        ltspice("1e999", element_value=True)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_percent_sign_is_an_error_to_26_and_a_hundredth_to_xvii(build: str, tmp_path: Path):
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_spice_value("8%")
    if rec.generation(build) == "xvii":
        assert rec.operating_point(build, "deck/suffix_percent", tmp_path)["v(n01)"] == (
            pytest.approx(0.08)
        )
    else:
        assert rec.entry(build, "deck/suffix_percent")["exit_code"] == 1


def flagged_as_milli(case_id: str) -> set[str]:
    return {
        token
        for finding in lint_input(case_id)
        if finding["rule_id"] == "suffix-mega-milli"
        for token in finding["evidence"]["tokens"]
    }


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_mega_or_milli_lint_flags_each_capital_m_ltspice_read_as_milli(
    build: str, tmp_path: Path
):
    """``1M`` is a milliohm, and so are ``1MHz`` and ``2M2``: the letters after
    the M do not turn it into mega. ``1MEG`` and ``1MIL`` are not milli and are
    left alone."""
    flagged: set[str] = set()
    read_as_milli: set[str] = set()
    for case_id in SUFFIX_DECKS:
        flagged |= flagged_as_milli(case_id)
        voltages = rec.operating_point(build, case_id, tmp_path / case_id.replace("/", "-"))
        for node, spelling in resistances(case_id).items():
            mantissa = spelling.partition("M")[0]
            if "M" in spelling and mantissa.replace(".", "").isdigit():
                milli = voltages[f"v({node})"] == pytest.approx(float(mantissa) * 1e-3, rel=1e-6)
                if milli or spelling == "2M2":  # 2M2 is 2.2 milli
                    read_as_milli.add(spelling)
    assert read_as_milli == {"1M", "1MHz", "2M2"}
    assert flagged == read_as_milli


# --------------------------------------------------------------------------
# Deck encoding
# --------------------------------------------------------------------------

#: Each micro-sign deck: the codec it is stored in, and whether a build that
#: decodes a deck as cp1252 still sees a micro sign in it.
MICRO_DECKS = {
    "deck/micro_cp1252": ("cp1252", True),
    "deck/micro_utf8": ("utf-8", False),
    "deck/micro_utf8_bom": ("utf-8-sig", False),
    "deck/micro_utf16le_bom": ("utf-16-le", True),
    "deck/micro_utf16le": ("utf-16-le", True),
    "deck/greek_mu_utf8": ("utf-8", False),
}


@pytest.mark.parametrize(("case_id", "codec"), [(k, v[0]) for k, v in MICRO_DECKS.items()])
def test_each_micro_sign_deck_is_detected_in_the_codec_it_is_stored_in(case_id: str, codec: str):
    data = (INPUTS / rec.CASES.case(case_id).source).read_bytes()
    text, detected = decode_spice_bytes_with_encoding(data)
    assert detected == codec
    (site,) = value_suffix_sites(lex(text).cards)
    assert site.micro
    assert site.token in (f"1{MICRO}", "1μ")


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(list(MICRO_DECKS))))
def test_a_cp1252_reader_loses_a_utf8_micro_sign_and_nothing_else(
    build: str, case_id: str, tmp_path: Path
):
    """LTspice XVII reads ``1µ`` stored as UTF-8 as 1: a million times too
    large, with no diagnostic. It reads cp1252 and UTF-16 correctly, and
    LTspice 26 reads every encoding correctly. The server folds the sign to
    ``u`` for exactly the builds it calls cp1252 readers."""
    manifest = rec.manifest(build)
    cp1252_reader = is_cp1252_ltspice_build(manifest["reported_build"])
    assert cp1252_reader == is_cp1252_ltspice_executable(manifest["executable"]["name"])
    assert cp1252_reader == (rec.generation(build) == "xvii")
    survives_cp1252 = MICRO_DECKS[case_id][1]
    expected = 1e-6 if survives_cp1252 or not cp1252_reader else 1.0
    assert rec.operating_point(build, case_id, tmp_path)["v(a)"] == pytest.approx(expected)
    # Not a line of the log says the suffix was dropped.
    log = rec.recorded(build, f"{case_id}.log").read_bytes().lower()
    assert b"error" not in log
    assert b"warning" not in log


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_ascii_u_the_server_folds_to_is_micro_to_every_build(build: str, tmp_path: Path):
    voltages = rec.operating_point(build, "deck/suffixes", tmp_path)
    written = {spelling: node for node, spelling in resistances("deck/suffixes").items()}
    assert voltages[f"v({written['1u']})"] == pytest.approx(1e-6)
    assert voltages[f"v({written['1U']})"] == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("build", "case_id"),
    list(rec.per_build(["deck/section_sign_utf8", "deck/section_sign_cp1252"])),
)
def test_a_section_sign_marks_where_an_instance_name_starts(
    build: str, case_id: str, tmp_path: Path
):
    """``R§Load`` is the resistor ``Load``: LTspice names its current
    ``I(Load)``, dropping the element letter and the marker, in either
    encoding of the deck."""
    traces = rec.operating_point(build, case_id, tmp_path)
    assert traces["v(a)"] == pytest.approx(1000.0)
    assert set(traces) == {"v(a)", "i(i1)", "i(load)"}


SECTION_SIGN_DECKS = ["deck/section_sign_utf8", "deck/section_sign_cp1252"]


def variant_values(text: str, path: Path, target: str, scratch: Path) -> dict[str, str]:
    """Each instance's value in the case deck one assignment of ``target`` writes."""
    circuit = CircuitDeck("dut", path, text)
    variation = AssignVariation(kind="assign", assign={target: ["2k"]})
    (case,) = materialize_variants(circuit, expand_variations([circuit], [variation]), scratch)
    return {
        line.ref: line.value or "" for line in map(read_instance, lex(case.text).cards) if line
    }


def drawn_for(text: str, path: Path, target: str, scratch: Path) -> list[str]:
    """The references a random component rule on ``target`` draws for."""
    circuit = CircuitDeck("dut", path, text)
    rule = {"rule": "component", "target": target, "tolerance": 0.1}
    variation = RandomVariation.model_validate(
        {"kind": "random", "runs": 1, "seed": 1, "rules": [rule]}
    )
    (case,) = materialize_variants(circuit, expand_variations([circuit], [variation]), scratch)
    prefix = "random:component:"
    return sorted(
        name.removeprefix(prefix) for name in case.assignments if name.startswith(prefix)
    )


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SECTION_SIGN_DECKS)))
@pytest.mark.parametrize("for_ngspice", [False, True], ids=["ltspice", "ngspice"])
def test_a_part_exported_with_a_section_sign_is_a_target_by_its_names(
    build: str, case_id: str, for_ngspice: bool, tmp_path: Path
):
    """A variation names ``R§Load`` as the sheet does (``Load``), as LTspice's
    results do (``I(Load)``, recorded) or as ``verify_circuit`` reports it
    (``RLoad``), and ``RLoad`` names it on either simulator's copy of the
    deck, though ngspice's has the marker taken out."""
    assert "i(load)" in rec.operating_point(build, case_id, tmp_path / "raw")
    export = tmp_path / "load.net"
    export.write_bytes((INPUTS / rec.CASES.case(case_id).source).read_bytes())
    _stem, data = deck_prep._read_export(export, for_ngspice=for_ngspice)
    text = decode_spice_bytes_with_encoding(data)[0]
    (load,) = [
        line.ref
        for line in map(read_instance, lex(text).cards)
        if line and line.nodes == ["a", "0"]
    ]
    names = ["RLoad"] if for_ngspice else ["R§Load", "RLoad", "Load", "load"]
    for index, target in enumerate(names):
        values = variant_values(text, export, target, tmp_path / str(index))
        assert values == {"I1": "1", load: "2k"}, target
        assert drawn_for(text, export, target, tmp_path / f"random{index}") == [load], target


def test_a_name_holding_two_section_signs_is_one_name_without_them(tmp_path: Path):
    """Every route that names an instance without its marker drops each ``§``:
    the ngspice copy of the deck, a variation's target on LTspice's copy, and
    the netlist comparison. Where LTspice itself puts a second ``§`` is not
    recorded; these are the server's own names, which must agree."""
    text = "* two signs\nV1 a 0 1\nR§a§b a 0 1k\n.op\n.end\n"
    export = tmp_path / "two.net"
    export.write_text(text, encoding="utf-8")
    _stem, data = deck_prep._read_export(export, for_ngspice=True)
    ngspice = decode_spice_bytes_with_encoding(data)[0]
    (ngspice_name,) = [
        line.ref for line in map(read_instance, lex(ngspice).cards) if line and line.ref != "V1"
    ]
    assert ngspice_name == "Rab"
    # The name the ngspice copy gives the part is a target on LTspice's copy.
    assert variant_values(text, export, ngspice_name, tmp_path / "assign") == {
        "V1": "1",
        "R§a§b": "2k",
    }
    assert drawn_for(text, export, ngspice_name, tmp_path / "random") == ["R§a§b"]
    # And the comparison pairs the two copies' parts by that name.
    comparison = compare_graphs(text, ngspice)
    assert comparison.equivalent, comparison
    assert not comparison.renamed, comparison


def test_a_name_two_parts_answer_to_is_refused(tmp_path: Path):
    """``RLoad`` is both a part and the other name of ``R§Load``, which an
    ngspice copy would also call ``RLoad``: no target can mean one of them."""
    text = deck_text("deck/section_sign_utf8").replace(".op", "RLoad a 0 3k\n.op")
    path = tmp_path / "load.cir"
    with pytest.raises(VariationError) as exc:
        variant_values(text, path, "RLoad", tmp_path / "out")
    assert exc.value.code == "ambiguous_target"
    with pytest.raises(VariationError) as exc:
        drawn_for(text, path, "RLoad", tmp_path / "random")
    assert exc.value.code == "ambiguous_target"
    assert variant_values(text, path, "Load", tmp_path / "load")["R§Load"] == "2k"
    assert drawn_for(text, path, "R*", tmp_path / "glob") == ["RLoad", "R§Load"]


#: The bytes cp1252 gives no character, each with the letter that follows it
#: in a node name of ``deck/bytes_in_node_names``.
UNDEFINED_IN_CP1252 = {0x81: "a", 0x8D: "b", 0x8F: "c", 0x90: "d", 0x9D: "e"}

#: Bytes cp1252 does define, which both builds read as the control character
#: of the same number all the same: an 8-bit deck is Latin-1 to LTspice. The
#: server reads the cp1252 character, so a name holding one is not the name
#: the build saved.
READ_AS_CP1252_BY_THE_SERVER_ONLY = {0x80: ("f", "€"), 0x93: ("g", "“")}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_byte_cp1252_leaves_undefined_is_read_as_each_build_reads_it(build: str, tmp_path: Path):
    """A deck that is not UTF-8 is read a byte at a time by both builds, each
    byte the character of the same number. XVII saves a node named with one;
    LTspice 26 takes no control character in a name and echoes the card it
    refuses, in UTF-8. Either way the name is on record, and for the five
    bytes cp1252 lacks the server reads the same one."""
    case_id = "deck/bytes_in_node_names"
    nodes: set[str] = set()
    for card in lex_deck(deck_text(case_id)).cards:
        line = read_instance(card) if card.kind == "instance" else None
        if line is not None:
            nodes.update(line.nodes)
    assert "�" not in "".join(nodes)
    if rec.generation(build) == "xvii":
        saved = set(rec.operating_point(build, case_id, tmp_path))

        def read_by_the_build(name: str) -> bool:
            return f"v({name})" in saved
    else:
        assert not ran(build, case_id)
        log = rec.recorded(build, f"{case_id}.log").read_bytes()
        assert b"Expected node name here." in log

        def read_by_the_build(name: str) -> bool:
            return f" {name} ".encode() in log

    for byte, letter in UNDEFINED_IN_CP1252.items():
        name = f"n{chr(byte)}{letter}"
        assert read_by_the_build(name)
        assert name in nodes
    for byte, (letter, as_cp1252) in READ_AS_CP1252_BY_THE_SERVER_ONLY.items():
        assert read_by_the_build(f"n{chr(byte)}{letter}")
        assert f"n{as_cp1252}{letter}" in nodes


def test_every_node_name_holding_a_control_byte_is_named_before_the_run():
    """Each node of ``deck/bytes_in_node_names`` holding a byte from 0x80 to
    0x9F is a lint warning, whichever of the two readings the server shows,
    and its ``saved_as`` is the name XVII saved (LTspice 26 refused them)."""
    findings = [
        finding
        for finding in lint_input("deck/bytes_in_node_names")
        if finding["rule_id"] == "node-control-byte-ltspice"
    ]
    saved = {finding["evidence"]["saved_as"] for finding in findings}
    expected = {
        *(f"n{chr(byte)}{letter}" for byte, letter in UNDEFINED_IN_CP1252.items()),
        *(
            f"n{chr(byte)}{letter}"
            for byte, (letter, _) in READ_AS_CP1252_BY_THE_SERVER_ONLY.items()
        ),
    }
    assert saved == {name.encode("unicode_escape").decode("ascii") for name in expected}
    assert {finding["severity"] for finding in findings} == {"warning"}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_byte_85_in_a_comment_ends_the_line_for_ltspice_26_alone(build: str, tmp_path: Path):
    """85 is an ellipsis in cp1252 and the next-line control in Latin-1, and
    it is half of many double-byte characters. LTspice 26 ends the comment
    there, and what follows is a card: the deck has a third resistor. XVII
    reads one comment. The server reads it as XVII does, so on LTspice 26 the
    circuit it describes is not the one that ran; the linter says so unless
    the session's LTspice is known to be XVII."""
    case_id = "deck/byte_85_in_comment"
    traces = set(rec.operating_point(build, case_id, tmp_path))
    assert ("i(r3)" in traces) == (rec.generation(build) != "xvii")
    cards = lex_deck(deck_text(case_id)).cards
    assert {card.name for card in cards if card.kind == "instance"} == {"V1", "R1", "R2"}
    xvii = rec.generation(build) == "xvii"
    findings = lint_input(
        case_id, cp1252_reader=rec.manifest(build)["reported_build"] if xvii else None
    )
    if xvii:
        assert findings == []
    else:
        (finding,) = findings
        assert finding["rule_id"] == "byte-85-ltspice"
        assert finding["at"]["line"] == 5
        assert finding["evidence"]["read_as_cards"] == ["R3 b 0 1k"]


# --------------------------------------------------------------------------
# Deck structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
class TestDeckStructure:
    """How a deck's lines are read. Each deck would give another voltage, or
    fail, were the line in question read the other way."""

    def voltage(self, build: str, case_id: str, scratch: Path, node: str = "a") -> float:
        return rec.operating_point(build, case_id, scratch)[f"v({node})"]

    def cards(self, case_id: str):
        return lex(deck_text(case_id)).cards

    @pytest.mark.parametrize("case_id", ["deck/comment_semicolon", "deck/comment_dollar"])
    def test_text_after_a_comment_character_is_not_the_value(
        self, build: str, case_id: str, tmp_path: Path
    ):
        assert self.voltage(build, case_id, tmp_path) == pytest.approx(1000.0)
        (resistor,) = [c for c in self.cards(case_id) if c.name == "R1"]
        line = read_instance(resistor)
        assert line is not None
        assert line.value == "1k"

    @pytest.mark.parametrize(
        "case_id", ["deck/comment_then_continuation", "deck/comment_in_model_parens"]
    )
    def test_a_comment_ends_at_the_end_of_its_line(self, build: str, case_id: str, tmp_path: Path):
        """The diode's ``n=2`` is on the ``+`` line after a ``;`` comment, which
        follows the parameters outside the card's parentheses or inside them.
        Read, it doubles the diode's drop at 1 mA from about 0.65 V to 1.31 V:
        the comment ends at the end of its line, and the card goes on."""
        assert self.voltage(build, case_id, tmp_path) == pytest.approx(1.3102, abs=1e-3)
        (model,) = [card for card in self.cards(case_id) if card.kind == "model"]
        assert ModelCard.from_card(model).params == {"is": "1e-14", "n": "2"}
        assert not [f for f in lint_input(case_id) if f["severity"] == "error"]

    @pytest.mark.parametrize(
        ("case_id", "node"),
        [("deck/dollar_inside_a_name", "N$1"), ("deck/dollar_leading_a_name", "$N_0001")],
    )
    def test_a_dollar_sign_in_a_word_is_part_of_the_word(
        self, build: str, case_id: str, node: str, tmp_path: Path
    ):
        """PSpice and OrCAD name nets ``N$1`` and ``$N_0001``. Both builds read
        each as a node: the source's ampere through 1k gives 1000 V on it. Only a
        ``$`` that stands alone begins a comment (``deck/comment_dollar``)."""
        assert self.voltage(build, case_id, tmp_path, node.lower()) == pytest.approx(1000.0)
        lines = {
            line.ref: line for line in map(read_instance, self.cards(case_id)) if line is not None
        }
        assert (lines["I1"].nodes, lines["R1"].nodes) == (["0", node], [node, "0"])
        assert (lines["I1"].value, lines["R1"].value) == ("1", "1k")
        assert not [f for f in lint_input(case_id) if f["severity"] == "error"]

    def test_the_first_line_is_a_title_even_when_it_reads_as_a_card(
        self, build: str, tmp_path: Path
    ):
        # Line 1 is "I1 0 a 5"; read as a card it would duplicate I1.
        assert self.voltage(build, "deck/title_card", tmp_path) == pytest.approx(1000.0)
        kept = lex_deck(deck_text("deck/title_card")).cards
        assert [card.body for card in kept if card.name == "I1"] == ["I1 0 a 1"]

    def test_the_first_line_is_a_title_even_when_it_reads_as_a_directive(self, build: str):
        """Line 1 is ``.param r=2k``. LTspice does not define ``r`` from it, so
        the resistor that uses ``{r}`` fails the run; the server's checks must
        not count the line as a directive either."""
        assert not ran(build, "deck/title_directive")
        kept = lex_deck(deck_text("deck/title_directive")).cards
        assert not [card for card in kept if card.kind == "param"]

    def test_cards_after_end_are_not_read(self, build: str, tmp_path: Path):
        # A second resistor after .end would halve the voltage.
        assert self.voltage(build, "deck/after_end", tmp_path) == pytest.approx(1000.0)
        (after,) = [c for c in self.cards("deck/after_end") if c.name == "R2"]
        assert after.trailing
        assert [c.name for c in lex_deck(deck_text("deck/after_end")).cards if c.name] == [
            "I1",
            "R1",
        ]

    def test_a_continuation_joins_across_a_comment_and_a_blank_line(
        self, build: str, tmp_path: Path
    ):
        assert self.voltage(build, "deck/continuation", tmp_path, "a") == pytest.approx(1000.0)
        assert self.voltage(build, "deck/continuation", tmp_path / "b", "b") == pytest.approx(
            2000.0
        )
        values = {
            line.ref: line.value
            for line in map(read_instance, self.cards("deck/continuation"))
            if line is not None
        }
        assert (values["R1"], values["I2"]) == ("1k", "1")

    @pytest.mark.parametrize("case_id", ["deck/crlf", "deck/no_end", "deck/case_insensitive"])
    def test_line_endings_a_missing_end_and_case_change_nothing(
        self, build: str, case_id: str, tmp_path: Path
    ):
        node = "node" if case_id == "deck/case_insensitive" else "a"
        assert self.voltage(build, case_id, tmp_path, node) == pytest.approx(1000.0)

    async def test_gnd_is_ground_at_the_top_level_and_inside_a_subcircuit(
        self, build: str, tmp_path: Path, state_no_sim, work_dir: Path
    ):
        """``R1 a gnd 1k`` carries the whole ampere to ground, and so does the
        same resistor inside a subcircuit: ``gnd`` is node 0 everywhere, except
        in a subcircuit that names a port ``gnd``, where it is that port."""
        voltages = rec.operating_point(build, "deck/gnd_alias", tmp_path)
        assert voltages["v(a)"] == pytest.approx(1000.0)
        assert voltages["v(b)"] == pytest.approx(1000.0)
        # Through the subcircuit's 1k to its gnd port, then 1k more to ground.
        assert voltages["v(c)"] == pytest.approx(2000.0)
        assert voltages["v(mid)"] == pytest.approx(1000.0)
        rows = await hierarchy_rows(state_no_sim, work_dir, "deck/gnd_alias")
        for instance in (("R1",), ("X1", "R1")):
            ground = rows[instance]["nodes"][1]
            # A node at the top level has no scope, which the reply leaves out.
            assert (ground.get("scope", []), ground["name"]) == ([], "0"), instance
        port = rows[("X2", "R1")]["nodes"][1]
        assert (port.get("scope", []), port["name"]) == ([], "mid")

    async def test_every_reader_of_a_deck_reads_a_port_named_gnd_as_the_port(
        self, build: str, tmp_path: Path, state_no_sim, work_dir: Path
    ):
        """Inside ``.subckt ported top gnd``, ``R1 top gnd`` goes to the port:
        the run puts v(c) at 2000, not 1000. So the comparison flattens the
        resistor onto the node the port is wired to; it matches the same deck
        with the port renamed, and not the deck with the resistor to node 0;
        and the floating- and dangling-node checks read the port as they read
        the renamed one."""
        voltages = rec.operating_point(build, "deck/gnd_alias", tmp_path)
        assert voltages["v(c)"] == pytest.approx(2000.0)
        text = deck_text("deck/gnd_alias")

        def body(header: str, card: str, deck: str = text) -> str:
            changed = deck.replace(".subckt ported top gnd\nR1 top gnd 1k", f"{header}\n{card}")
            assert changed != deck
            return changed

        renamed = body(".subckt ported top p", "R1 top p 1k")
        to_ground = body(".subckt ported top gnd", "R1 top 0 1k")

        nodes = {
            c.ref.lower(): c.nodes for c in flatten_graph(parse_netlist_graph(text)).components
        }
        assert nodes["x2.r1"] == ("c", "mid")
        assert nodes["x1.r1"] == ("b", "0")
        assert nodes["r1"] == ("a", "0")
        assert compare_graphs(text, renamed).equivalent
        assert not compare_graphs(text, to_ground).equivalent
        path = work_dir / "gnd_alias.cir"
        path.write_text(text, encoding="utf-8")
        for reference, equivalent in ((renamed, True), (to_ground, False)):
            verified = await handle_verify_circuit(
                VerifyCircuitInput.model_validate(
                    {"path": str(path), "checks": ["compare"], "compare": {"reference": reference}}
                ),
                state_no_sim,
            )
            assert verified.structured_content is not None
            assert verified.structured_content["comparison"]["equivalent"] is equivalent

        # With its only path to ground a capacitor, the node the port is wired
        # to floats, whatever the port is called.
        floating = text.replace("R3 mid 0 1k", "C3 mid 0 1n")
        floating_renamed = body(".subckt ported top p", "R1 top p 1k", floating)
        found = validate_netlist_bias_topology(lex_deck(floating).cards)
        assert found
        assert [i["message"] for i in found] == [
            i["message"] for i in validate_netlist_bias_topology(lex_deck(floating_renamed).cards)
        ]
        # A port wired to nothing in its body is reported whatever it is called.
        unused = body(".subckt ported top gnd", "R1 top 0 1k")
        unused_renamed = body(".subckt ported top p", "R1 top 0 1k")
        dangling = [
            str(i["message"]).replace("'gnd'", "'p'")
            for i in validate_netlist_dangling_nodes(lex_deck(unused).cards)
        ]
        assert dangling
        assert dangling == [
            i["message"] for i in validate_netlist_dangling_nodes(lex_deck(unused_renamed).cards)
        ]

    @pytest.mark.parametrize("at", ["0", "gnd", "GND"])
    async def test_the_net_query_reads_gnd_as_node_0(
        self, build: str, at: str, tmp_path: Path, state_no_sim, work_dir: Path
    ):
        """The run's reading: ``R1 a gnd`` and ``R2 gnd 0`` are on ground, so a
        query for ground by any of its spellings lists them with node 0's
        cards. Only the deck's own cards are listed, not those in a subcircuit
        body, and the reply says ngspice's kiltpsa mode may read ``gnd``
        otherwise."""
        voltages = rec.operating_point(build, "deck/gnd_alias", tmp_path)
        assert voltages["v(a)"] == pytest.approx(1000.0)
        path = work_dir / "gnd_alias.cir"
        path.write_bytes((INPUTS / rec.CASES.case("deck/gnd_alias").source).read_bytes())
        (result,) = await _run(state_no_sim, [{"kind": "net", "path": str(path), "at": at}])
        assert result["ok"], result
        data = result["data"]
        assert {(m["reference"], m["terminal"]) for m in data["members"]} == {
            ("I1", 1),
            ("R1", 2),
            ("R2", 1),
            ("R2", 2),
            ("I2", 1),
            ("I3", 1),
            ("R3", 2),
        }
        assert any("kiltpsa" in warning for warning in data.get("warnings", [])), data


#: Recorded decks whose lines each build reads in a way a line-by-line reader
#: would not: a ``+`` line continuing a card past a comment and past a blank
#: line, and a lone ``$`` starting a comment after a value.
LINE_FORM_DECKS = ["deck/continuation", "deck/comment_dollar", "deck/comment_semicolon"]


def as_voltage_sources(text: str) -> str:
    """``text`` with every source and resistor written as a voltage source
    (``I1`` is ``VI1``), so the reader of a deck's voltage sources meets each
    line form the recording holds."""
    return re.sub(r"(?m)^([IR])", r"V\1", text)


@pytest.mark.parametrize("case_id", LINE_FORM_DECKS)
def test_the_source_amplitudes_are_the_values_the_lexer_reads(case_id: str):
    """The result observations' largest-source reading takes each voltage
    source's value from the card LTspice reads (``TestDeckStructure`` holds
    the lexer to the recordings)."""
    text = as_voltage_sources(deck_text(case_id))
    lines = [line for line in map(read_instance, lex_deck(text).cards) if line is not None]
    assert parse_source_amplitudes(text) == {
        line.ref: parse_spice_value(line.value or "") for line in lines
    }
    assert {line.ref for line in lines} >= {"VI1", "VR1"}


def test_the_requested_outputs_are_the_cards_the_lexer_reads():
    """A ``.four`` and a ``.meas`` continued past a comment line, as the
    recorded ``deck/continuation`` continues a card, and a ``;`` comment."""
    text = as_voltage_sources(deck_text("deck/continuation")).replace(
        ".op\n",
        ".tran 1m\n"
        ".four 1k V(a) ; the input\n"
        "* and the output\n"
        "+ V(b)\n"
        ".meas tran\n"
        "* the peak\n"
        "+ vmax MAX V(a)\n",
    )
    assert parse_requested_outputs(text) == {"meas": ["vmax"], "four": ["V(a)", "V(b)"]}


async def hierarchy_rows(state, work_dir: Path, case_id: str) -> dict[tuple[str, ...], dict]:
    """The server's resolved view of a recorded deck, by instance path."""
    path = work_dir / Path(rec.CASES.case(case_id).source).name
    path.write_bytes((INPUTS / rec.CASES.case(case_id).source).read_bytes())
    (result,) = await _run(
        state, [{"kind": "hierarchy", "path": str(path), "simulator": "ltspice"}]
    )
    assert result["ok"], result
    return {tuple(row["instance"]): row for row in result["data"]["instances"]}


# --------------------------------------------------------------------------
# Cards a deck's readers read
# --------------------------------------------------------------------------

#: Decks with a card LTspice does not read: a title that reads as a card or as
#: a directive, and a card after ``.end`` (``TestDeckStructure`` holds what
#: each build did with them).
UNREAD_CARD_DECKS = ["deck/title_card", "deck/title_directive", "deck/after_end"]


def as_ltspice_reads(text: str) -> str:
    """``text`` holding only the cards LTspice reads: a comment for its title
    line, and nothing after ``.end``."""
    lines = ["* title\n", *text.splitlines(keepends=True)[1:]]
    end = next(i for i, line in enumerate(lines) if line.strip().casefold() == ".end")
    return "".join(lines[: end + 1])


def no_structural_change(delta: dict[str, list]) -> bool:
    return not any(changes for key, changes in delta.items() if key != "components_renamed")


@pytest.mark.parametrize("case_id", UNREAD_CARD_DECKS)
def test_an_edit_may_change_only_the_cards_the_deck_reading_reads(case_id: str):
    """``lex_for_edit`` keeps every line of the deck for ``emit``, and offers
    an edit the cards ``lex_deck`` reads, as the same objects, so a change made
    through them is written. ``test_recorded_ltspice_unread_cards.py`` holds
    each edit to it."""
    text = deck_text(case_id)
    lexed = lex_for_edit(text, deck=True)
    assert emit(lexed.cards) == text
    assert [card.body for card in lexed.read] == [card.body for card in lex_deck(text).cards]
    assert all(lexed.reads(card) for card in lexed.read)
    assert {id(card) for card in lexed.read} <= {id(card) for card in lexed.cards}


@pytest.mark.parametrize("case_id", UNREAD_CARD_DECKS)
def test_the_comparisons_read_only_the_cards_ltspice_reads(case_id: str):
    """Both compare modes call a deck the same circuit as the one LTspice
    reads from it: neither counts the title as a second ``I1`` or a
    ``.param``, nor the resistor after ``.end``."""
    text = deck_text(case_id)
    read = as_ltspice_reads(text)
    graphs = compare_graphs(text, read)
    assert graphs.equivalent, graphs.as_dict()
    assert no_structural_change(structural_delta(read_deck(text), read_deck(read)))


@pytest.mark.parametrize("case_id", UNREAD_CARD_DECKS)
async def test_the_net_query_lists_only_cards_ltspice_reads(
    case_id: str, state_no_sim, work_dir: Path
):
    answers = []
    text = deck_text(case_id)
    for name, deck in (("deck.cir", text), ("read.cir", as_ltspice_reads(text))):
        path = work_dir / name
        path.write_text(deck, encoding="utf-8")
        (result,) = await _run(state_no_sim, [{"kind": "net", "path": str(path), "at": "a"}])
        assert result["ok"], result
        answers.append(result["data"]["members"])
    assert answers[0] == answers[1]


@pytest.mark.parametrize("case_id", UNREAD_CARD_DECKS)
async def test_the_checks_read_only_the_cards_ltspice_reads(
    case_id: str, state_no_sim, work_dir: Path
):
    """The linter, verify_circuit's syntax and quality checks and the run's
    requirements see the deck LTspice reads."""

    async def readings(name: str, text: str) -> tuple:
        path = work_dir / name
        path.write_text(text, encoding="utf-8")
        verified = await handle_verify_circuit(
            VerifyCircuitInput.model_validate({"path": str(path)}), state_no_sim
        )
        assert verified.structured_content is not None
        linted = lint_deck(text, path, "ltspice", "LTspice")
        return (
            sorted((f["rule_id"], f["at"]["line"]) for f in linted),
            sorted(
                (f["rule_id"], f["at"].get("line"))
                for f in verified.structured_content.get("findings", [])
            ),
            deck_requirements(path),
            parse_requested_outputs(text),
            parse_source_amplitudes(text),
        )

    text = deck_text(case_id)
    assert await readings("deck.cir", text) == await readings("read.cir", as_ltspice_reads(text))


#: The cards ``deck/after_end`` and ``deck/title_directive`` show LTspice does
#: not read, written as cards each reader would otherwise act on: a second
#: analysis and a one-node resistor after ``.end``, a ``.tran``, a refused
#: ``.meas`` and a voltage source as the title.
UNREAD_CARDS = [
    pytest.param(
        ".tran 1m\n"
        "V1 a 0 1\n"
        "R1 a 0 1k\n"
        ".meas tran top MAX V(a)\n"
        ".op\n"
        ".end\n"
        ".ac dec 10 1 1k\n"
        "R9 lonely\n"
        "V9 b 0 50\n"
        ".meas ac g FIND vdb(a) AT 1k\n"
        ".four 1k V(a)\n",
        id="analysis-title-and-cards-after-end",
    ),
    pytest.param(
        "V7 c 0 70\nV1 a 0 1\nR1 a 0 1k\n.meas tran top MAX V(a)\n.op\n.end\n",
        id="source-title",
    ),
    pytest.param(
        ".meas ac g FIND vdb(a) AT 1k\nV1 a 0 1\nR1 a 0 1k\n.op\n.end\n",
        id="refused-measurement-title",
    ),
]


@pytest.mark.parametrize("text", UNREAD_CARDS)
async def test_a_card_ltspice_does_not_read_changes_no_reading(
    text: str, state_no_sim, work_dir: Path
):
    """Each reader gives the same answer for the deck as for the deck LTspice
    reads from it, and so refuses nothing on a card it does not read."""
    path = work_dir / "deck.cir"
    path.write_text(text, encoding="utf-8")
    read = work_dir / "read.cir"
    read.write_text(as_ltspice_reads(text), encoding="utf-8")
    assert lint_deck(text, path, "ltspice", "LTspice") == []
    verified = await handle_verify_circuit(
        VerifyCircuitInput.model_validate({"path": str(path)}), state_no_sim
    )
    assert verified.structured_content is not None
    assert [
        f for f in verified.structured_content.get("findings", []) if f["severity"] == "error"
    ] == []
    assert deck_requirements(path) == deck_requirements(read)
    assert parse_requested_outputs(text) == parse_requested_outputs(as_ltspice_reads(text))
    assert parse_source_amplitudes(text) == parse_source_amplitudes(as_ltspice_reads(text))


# --------------------------------------------------------------------------
# Card forms
# --------------------------------------------------------------------------

#: The lint and arity findings that refuse a deck, per recorded deck. A deck
#: not listed must come through both without one.
REFUSED = {
    # Both builds refuse the run.
    "deck/keyed_c": {"value-keyword-ltspice"},
    "deck/keyed_l": {"value-keyword-ltspice"},
    "deck/default_models": {"model-missing"},
    # LTspice runs it and the measurement fails for want of the unsaved node.
    "deck/save_omits_meas": {"save-meas-coverage"},
    # "More than one analysis specified."
    "deck/ac_and_tran": {"analysis-count-ltspice"},
    # Functions only LTspice's waveform viewer has.
    "deck/meas_function_vdb": {"meas-function-ltspice"},
    "deck/meas_function_phase": {"meas-function-ltspice"},
    "deck/meas_function_group_delay": {"meas-function-ltspice"},
}

FORMS = sorted(rec.cases_of("deck-forms"))


def refusals(case_id: str) -> set[str]:
    text = deck_text(case_id)
    lint = {f["rule_id"] for f in lint_input(case_id) if f["severity"] == "error"}
    arity = {
        str(issue["check"])
        for issue in validate_netlist_arity(lex_deck(text).cards, simulator="LTspice")
        if issue["severity"] == "error"
    }
    return lint | arity


@pytest.mark.parametrize("case_id", FORMS)
def test_the_checks_refuse_exactly_the_forms_listed(case_id: str):
    assert refusals(case_id) == REFUSED.get(case_id, set())


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(FORMS)))
def test_a_form_the_checks_accept_ran_on_ltspice(build: str, case_id: str):
    """B-source ``R=`` and ``P=``, a spaced B expression, comma-separated
    ``IC=`` values, ``Q=`` and ``Flux=``, a value keyed ``R=``, ``params:`` on a
    subcircuit, trailing area factors, the ``.tran`` shorthands, ``.op`` beside
    ``.tran``."""
    if case_id in REFUSED:
        pytest.skip("not a form the checks accept")
    assert ran(build, case_id)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_two_analyses_in_one_deck_are_refused_by_ltspice(build: str):
    assert not ran(build, "deck/ac_and_tran")
    assert ran(build, "deck/op_and_tran")


@pytest.mark.parametrize("build", rec.BUILDS)
class TestRecordedValuesOfTheForms:
    """The forms do what the card says, not merely run."""

    def test_a_behavioural_resistor_and_power_sink(self, build: str, tmp_path: Path):
        assert rec.operating_point(build, "deck/bsource_r", tmp_path / "r")["v(a)"] == (
            pytest.approx(1000.0)
        )
        sink = rec.operating_point(build, "deck/bsource_p", tmp_path / "p")
        # One watt drawn from the node: V(b) times the current into B1.
        assert sink["v(b)"] * sink["i(b1)"] == pytest.approx(1.0, rel=1e-5)

    def test_a_spaced_expression_is_read_whole(self, build: str, tmp_path: Path):
        # "V = V(b) + 1" with V(b) = 2.
        assert rec.operating_point(build, "deck/bsource_spaced", tmp_path)["v(a)"] == (
            pytest.approx(3.0)
        )

    def test_a_value_keyed_as_r_is_the_resistance(self, build: str, tmp_path: Path):
        assert rec.operating_point(build, "deck/keyed_r", tmp_path)["v(a)"] == pytest.approx(
            1000.0
        )

    def test_params_on_a_subcircuit_call_overrides_with_or_without_the_keyword(
        self, build: str, tmp_path: Path
    ):
        voltages = rec.operating_point(build, "deck/params_keyword", tmp_path)
        # 3 V across r over 1k: r=2k gives 1 V, the 1k default 1.5 V.
        assert voltages["v(mid)"] == pytest.approx(1.0)
        assert voltages["v(mid2)"] == pytest.approx(1.0)
        assert voltages["v(mid3)"] == pytest.approx(1.5)


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_charge_or_flux_defined_part_has_two_terminals(
    build: str, tmp_path: Path, state_no_sim, work_dir: Path
):
    """``C1 b 0 Q=100n*x`` and ``L1 c 0 Flux=1m*x``, each behind 1k from a
    1 V step: current flows through each to ground, so the node it sits on
    stays below the step where both builds measured it. Every reader of the
    card's terminals gives both: the graph, the inspect net query and the
    validator."""
    logs = rec.decode_log(build, "deck/keyword_values", tmp_path)
    measured = logs.value("measurements")["measurements"]
    # With no path to ground each node would sit at the step's 1 V.
    assert 0.9 < measured["vb"]["values"][0] < 0.999
    assert 0.6 < measured["vc"]["values"][0] < 0.7

    text = deck_text("deck/keyword_values")
    expected = {"C1": ["b", "0"], "L1": ["c", "0"]}
    graph = {c.ref: list(c.nodes) for c in parse_netlist_graph(text).components}
    assert {ref: graph[ref] for ref in expected} == expected
    lines = {line.ref: line for line in map(read_instance, lex_deck(text).cards) if line}
    assert {ref: _instance_terminals(lines[ref])[0] for ref in expected} == expected
    assert {ref: lines[ref].nodes for ref in expected} == expected
    path = work_dir / "keyword_values.cir"
    path.write_text(text, encoding="utf-8")
    (result,) = await _run(state_no_sim, [{"kind": "net", "path": str(path), "at": "0"}])
    assert result["ok"], result
    on_ground = {(m["reference"], m["terminal"]) for m in result["data"]["members"]}
    assert {("C1", 2), ("L1", 2)} <= on_ground


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("function", ["vdb", "phase", "group_delay"])
def test_a_function_the_directive_check_refuses_fails_in_ltspice(build: str, function: str):
    """``vdb()``, ``phase()`` and ``group_delay()`` are not functions a
    ``.meas`` can call. LTspice 26 stops at the directive and takes none of
    the deck's measurements; XVII fails that one and takes the others."""
    case_id = f"deck/meas_function_{function}"
    directive = next(line for line in deck_text(case_id).splitlines() if " asked " in line)
    error = validate_directive(directive, "LTspice")
    assert error is not None
    assert error.rule_name == f"{function}_in_meas"
    log = read_spice_text(rec.recorded(build, f"{case_id}.log"))
    if rec.generation(build) == "xvii":
        assert 'Measurement "asked" FAIL\'ed' in log
        assert "before:" in log
        assert "after:" in log
    else:
        assert "No such function defined." in log
        assert "before:" not in log
        assert "after:" not in log


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_functions_the_directive_check_allows_all_measure(build: str):
    text = deck_text("deck/meas_functions")
    for line in text.splitlines():
        if line.startswith(".meas"):
            assert validate_directive(line, "LTspice") is None, line
    log = read_spice_text(rec.recorded(build, "deck/meas_functions.log"))
    for name in ("g_mag", "g_ph", "g_re", "g_im", "g_db"):
        assert f"{name}:" in log
    assert "FAIL" not in log


# --------------------------------------------------------------------------
# A .meas of a node pair, against what .save names
# --------------------------------------------------------------------------

#: V(in,mid) at 0.9 ms: C1 charges through 2k (tau 200 us), and R1 drops half.
PAIR_VOLTS = math.exp(-4.5) / 2


def pair_run(build: str, case_id: str, tmp_path: Path) -> tuple[list[str], MeasurementsOutput]:
    """The traces a recorded pair run's raw holds, and its log's measurements."""
    traces = rec.decode(build, case_id, tmp_path).raw.get_trace_names()
    return traces, parse_measurements(rec.recorded(build, f"{case_id}.log"))


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_pair_measured_from_its_two_saved_nodes_passes_the_lint(build: str, tmp_path: Path):
    """``.save V(in) V(mid)`` with ``FIND V(in,mid)``: both builds take the
    measurement from the two nodes, the only voltages the raw holds, and
    ``save-meas-coverage`` accepts the deck, reading the pair as its nodes."""
    traces, data = pair_run(build, "deck/save_pair_both_nodes", tmp_path)
    assert traces == ["time", "V(in)", "V(mid)"]
    assert data["measurements"]["vpair"]["values"][0] == pytest.approx(PAIR_VOLTS, abs=5e-4)
    assert refusals("deck/save_pair_both_nodes") == set()


@pytest.mark.parametrize("build", rec.BUILDS)
def test_saving_a_pair_saves_its_two_nodes(build: str, tmp_path: Path):
    """``.save V(in,mid)`` is accepted: the raw holds V(in) and V(mid), not the
    pair, the pair is measured from them as when both are saved, and the
    checks accept the deck."""
    traces, data = pair_run(build, "deck/save_pair_as_pair", tmp_path)
    assert traces == ["time", "V(in)", "V(mid)"]
    _, both = pair_run(build, "deck/save_pair_both_nodes", tmp_path)
    assert data["measurements"]["vpair"]["values"] == both["measurements"]["vpair"]["values"]
    assert refusals("deck/save_pair_as_pair") == set()


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_node_against_ground_is_not_measured_with_only_the_node_saved(
    build: str, tmp_path: Path
):
    """``.save V(mid)`` with ``FIND V(mid,0)``: neither build takes the
    measurement. LTspice 26 stops it with "Voltage not found." and XVII prints
    it FAIL'ed. The raw holds V(mid) alone, and ``save-meas-coverage`` refuses
    the deck."""
    traces, data = pair_run(build, "deck/save_pair_to_ground", tmp_path)
    assert traces == ["time", "V(mid)"]
    if rec.generation(build) == "xvii":
        assert data["failed_measurements"] == ["vground"]
    else:
        assert "vground" not in data["measurements"]
        log = read_spice_text(rec.recorded(build, "deck/save_pair_to_ground.log"))
        assert "Voltage not found." in log
    assert refusals("deck/save_pair_to_ground") == {"save-meas-coverage"}


# --------------------------------------------------------------------------
# A mismatch draw on a MOSFET inside a subcircuit
# --------------------------------------------------------------------------


def source_currents(build: str, case_id: str, tmp_path: Path) -> dict[str, float]:
    """The current through each 0 V source of a recorded operating point, by name."""
    values = rec.operating_point(build, case_id, tmp_path / case_id.replace("/", "-"))
    return {name[2:-1]: value for name, value in values.items() if name.startswith("i(v")}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_delvto_on_a_bsim4_device_is_its_signed_vth0_shifted(build: str, tmp_path: Path):
    """``delvto=0.1`` on a BSIM4 device draws the current of a model with
    ``vth0`` 0.1 V higher. On a p-channel device that is ``vth0`` -0.4 for -0.5,
    the signed shift ngspice also makes, not -0.6."""
    i = source_currents(build, "deck/mos_delvto_bsim4", tmp_path)
    assert i["vn1"] == pytest.approx(i["vn2"], rel=1e-9)
    assert i["vn1"] < i["vn0"]
    assert i["vp1"] == pytest.approx(i["vp2"], rel=1e-9)
    assert i["vp1"] != pytest.approx(i["vp3"], rel=1e-3)


#: Decks giving a device an instance parameter LTspice does not take, each
#: device with it paired with its twin without it.
UNKNOWN_INSTANCE_PARAMETER = {
    "deck/mos_mulu0_bsim4": [("v1", "v0")],
    "deck/mos_delvto_bsim3": [("vn1", "vn0"), ("vp1", "vp0")],
    "deck/mos_mulu0_bsim3": [("v1", "v0")],
}


@pytest.mark.parametrize(
    ("build", "case_id"), [(b, c) for b in rec.BUILDS for c in UNKNOWN_INSTANCE_PARAMETER]
)
def test_an_instance_parameter_ltspice_lacks_stops_26_and_xvii_runs_without_it(
    build: str, case_id: str, tmp_path: Path
):
    """``mulu0`` on BSIM4 and BSIM3, and ``delvto`` on BSIM3, are unknown to
    both builds. LTspice 26 stops on "Unknown parameter"; XVII prints the same
    error and runs on with the parameter dropped, so the device draws its twin's
    current."""
    assert "Unknown parameter" in read_spice_text(rec.recorded(build, f"{case_id}.log"))
    if rec.generation(build) == "xvii":
        i = source_currents(build, case_id, tmp_path)
        for given, twin in UNKNOWN_INSTANCE_PARAMETER[case_id]:
            assert i[given] == i[twin]
    else:
        assert rec.entry(build, case_id)["exit_code"] == 1
        assert not rec.has(build, f"{case_id}.raw")


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_model_copied_with_vth0_or_u0_changed_takes_effect(build: str, tmp_path: Path):
    """On BSIM4 and BSIM3 alike, a model with ``u0`` scaled by 1.2 draws more
    current than its original, and one with ``vth0`` 0.1 V higher draws less."""
    i = source_currents(build, "deck/mos_model_variants", tmp_path)
    assert i["vb"] > i["va"]
    assert i["vd"] < i["vc"] < i["ve"]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_subcircuit_copy_uses_the_model_copy_it_names(build: str, tmp_path: Path):
    """A copy of a subcircuit that carries a copy of its local model, and one
    that names a copy of a top-level model, each draw the shifted model's
    current, while the originals draw the unshifted one."""
    i = source_currents(build, "deck/subckt_model_copy", tmp_path)
    reference = source_currents(build, "deck/mos_delvto_bsim4", tmp_path)
    assert i["va"] == i["vc"] == pytest.approx(reference["vn0"], rel=1e-9)
    assert i["vb"] == i["vd"] == pytest.approx(reference["vn2"], rel=1e-9)


# --------------------------------------------------------------------------
# What a deck means
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
class TestDeckSemantics:
    def test_a_parameter_named_temp_never_sets_the_temperature(self, build: str, tmp_path: Path):
        """LTspice 26 refuses ``.param temp=50``; XVII runs the deck at 27
        degrees as if the line were not there. The lint refuses it for both."""
        findings = lint_input("deck/param_temp")
        assert [f["rule_id"] for f in findings if f["severity"] == "error"] == ["temp-as-param"]
        if rec.generation(build) == "xvii":
            # 1 V across 1k with tc1=0.01: 1 mA only at the nominal 27 degrees.
            current = rec.operating_point(build, "deck/param_temp", tmp_path)["i(r1)"]
            assert current == pytest.approx(1e-3)
        else:
            log = read_spice_text(rec.recorded(build, "deck/param_temp.log"))
            assert '"temp" is a reserved symbol for temperature' in log
        hot = rec.operating_point(build, "deck/options_temp", tmp_path / "hot")["i(r1)"]
        assert hot == pytest.approx(1 / (1000 * (1 + 0.01 * (50 - 27))), rel=1e-5)

    async def test_the_caret_is_not_a_power_and_the_server_does_not_guess_it(
        self, build: str, tmp_path: Path, state_no_sim, work_dir: Path
    ):
        """``2^3^2`` is 1 and ``(2^3)+5`` is 5: the caret is exclusive-or of
        two true values. ``2**3`` is 8 and ``2**3**2`` is 512."""
        read = rec.operating_point(build, "deck/caret_power", tmp_path)
        assert [read[f"v({node})"] for node in "abcd"] == pytest.approx([1, 8, 512, 5])
        rows = await hierarchy_rows(state_no_sim, work_dir, "deck/caret_power")
        for reference, node in (("R1", "a"), ("R2", "b"), ("R3", "c"), ("R4", "d")):
            fact = rows[(reference,)]["value"]
            if fact.get("value") is not None:
                assert fact["value"] == pytest.approx(read[f"v({node})"]), reference
        # The two the server cannot be wrong about because it declines them:
        # unresolved, with no value in the reply.
        for reference in ("R1", "R4"):
            assert rows[(reference,)]["value"]["status"] == "unresolved"
            assert "value" not in rows[(reference,)]["value"]
        with pytest.raises(ValueError, match="caret"):
            evaluate("2^3", lambda name: 0.0, simulator="ltspice")

    async def test_a_subcircuits_header_default_beats_a_param_in_its_body(
        self, build: str, tmp_path: Path, state_no_sim, work_dir: Path
    ):
        read = rec.operating_point(build, "deck/subckt_param_precedence", tmp_path)
        rows = await hierarchy_rows(state_no_sim, work_dir, "deck/subckt_param_precedence")
        for instance, node, ohms in (
            (("X1", "R1"), "a", 2000.0),  # the header's 2k, not the body's 3k
            (("X2", "R1"), "b", 7000.0),  # the call's own value
            (("X3", "X1", "R1"), "c", 2000.0),  # a value given to the wrapper stops there
        ):
            assert read[f"v({node})"] == pytest.approx(ohms)
            assert rows[instance]["value"]["value"] == pytest.approx(ohms), instance


# --------------------------------------------------------------------------
# Includes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
def test_an_include_beside_the_deck_is_found(build: str, tmp_path: Path):
    assert rec.operating_point(build, "deck/include_relative", tmp_path)["v(a)"] == (
        pytest.approx(3000.0)
    )


@pytest.mark.parametrize("build", rec.BUILDS)
def test_ltspice_reads_a_lib_section_name_as_part_of_the_file_name(build: str):
    """``.lib corners.lib tt`` selects a section in ngspice. LTspice has no
    sections: it looks for a file called ``corners.lib tt`` and stops. The lint
    refuses the deck before the run."""
    assert not ran(build, "deck/lib_section")
    log = read_spice_text(rec.recorded(build, "deck/lib_section.log"))
    if rec.generation(build) == "xvii":
        assert 'Could not open library file "corners.lib tt"' in log
    else:
        assert "File not found." in log
        assert ".lib corners.lib tt" in log
    findings = lint_input("deck/lib_section")
    assert [f["rule_id"] for f in findings if f["severity"] == "error"] == ["lib-section-ltspice"]


# --------------------------------------------------------------------------
# Three-node transistors
# --------------------------------------------------------------------------


def waves(build: str, case_id: str, scratch: Path) -> dict[str, np.ndarray]:
    """Every trace of a recorded run, by lower-cased name."""
    raw = rec.decode(build, case_id, scratch).raw
    return {name.lower(): np.asarray(raw.get_wave(name, 0)) for name in raw.get_trace_names()}


def same_wave(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, rtol=1e-9, atol=0))


def card_of(text: str, ref: str) -> str:
    """The top-level card named ``ref``: the cards before the first ``.subckt``."""
    top = re.split(r"(?im)^\.subckt", text, maxsplit=1)[0]
    (card,) = [line for line in top.splitlines() if line.split()[:1] == [ref]]
    return card


def with_card(text: str, old: str, new: str) -> str:
    lines = text.splitlines()
    assert lines.count(old) == 1, old
    return "\n".join(new if line == old else line for line in lines) + "\n"


def moved_to(card: str, stage: str) -> str:
    """A twin stage's card, renamed and rewired onto ``stage``'s nodes:
    ``Q3 c3 b3 e3 e3 QN`` onto stage 1 is ``Q1 c1 b1 e1 e1 QN``."""
    ref, *rest = card.split()
    twin = ref[1:]
    nodes = (re.sub(rf"^([a-z]+){twin}$", rf"\g<1>{stage}", token) for token in rest)
    return " ".join((f"{ref[0]}{stage}", *nodes))


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_comparison_reads_a_three_node_q_card_as_ltspice_runs_it(build: str, tmp_path: Path):
    """A three-node Q card compares equal to a four-node card exactly when
    LTspice's run of the two is the same, which is with the substrate at
    ground and not at the emitter or the collector: NPN, PNP, and inside a
    subcircuit."""
    case_id = "deck/bjt_three_node"
    read = waves(build, case_id, tmp_path)
    text = deck_text(case_id)
    matched: list[str] = []
    for stage, twins in (("1", "234"), ("5", "678")):
        three = card_of(text, f"Q{stage}")
        for twin in twins:
            four = moved_to(card_of(text, f"Q{twin}"), stage)
            same = all(same_wave(read[f"v({n}{twin})"], read[f"v({n}{stage})"]) for n in "cbe")
            result = compare_graphs(text, with_card(text, three, four))
            assert result.equivalent == same, (four, result.as_dict())
            if same:
                matched.append(four)
    assert matched == ["Q1 c1 b1 e1 0 QN", "Q5 c5 b5 e5 0 QP"]

    assert same_wave(read["v(c9)"], read["v(c10)"])
    assert compare_graphs(text, with_card(text, "Q1 c b e QN", "Q1 c b e 0 QN")).equivalent


@pytest.mark.parametrize("build", rec.BUILDS)
def test_three_node_q_cards_match_the_export_of_three_pin_transistors(build: str):
    """The export writes a three-pin transistor's substrate as 0, so a
    reference that leaves it out describes the same sheet."""
    export = rec.export_text(build, "export/boilerplate")
    written: list[str] = []
    for line in export.splitlines():
        tokens = line.split()
        if tokens[:1] and tokens[0][0] == "Q":
            assert tokens[4] == "0", line
            line = " ".join(tokens[:4] + tokens[5:])
        written.append(line)
    assert compare_graphs("\n".join(written) + "\n", export).equivalent


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_three_node_m_card_is_not_the_card_an_export_writes(build: str, tmp_path: Path):
    """With a level-1 model LTspice 26 refuses a three-node M card and XVII
    runs it with the bulk at ground, where the export puts the bulk at the
    source. The comparison reports the arity and says what to write."""
    case_id = "deck/mos_three_node"
    if rec.generation(build) == "xvii":
        read = waves(build, case_id, tmp_path)
        for stage, at_source, at_ground in (("1", "2", "3"), ("4", "5", "6")):
            assert same_wave(read[f"v(s{stage})"], read[f"v(s{at_ground})"])
            assert not same_wave(read[f"v(s{stage})"], read[f"v(s{at_source})"])
    else:
        assert not ran(build, case_id)
        log = read_spice_text(rec.recorded(build, f"{case_id}.log"))
        assert log.count("Node or model name expected.") == 2
        assert "M1 vdd gn s1 NM" in log
        assert "M4 vss gp s4 PM" in log
    text = deck_text(case_id)
    result = compare_graphs(text, with_card(text, "M1 vdd gn s1 NM", "M1 vdd gn s1 s1 NM"))
    assert not result.equivalent
    (error,) = result.arity_errors
    assert (error.ref, error.reference_arity, error.candidate_arity) == ("M1", 3, 4)
    assert "Write the bulk as the fourth node, at the source" in error.detail


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_vdmos_card_runs_the_same_with_three_nodes_or_its_bulk_at_its_source(
    build: str, tmp_path: Path
):
    """What the arity detail says of VDMOS: three nodes are the bulk at the
    source, and a bulk anywhere else is refused."""
    read = waves(build, "deck/vdmos_three_node", tmp_path)
    for node in "dgs":
        assert same_wave(read[f"v({node}1)"], read[f"v({node}2)"]), node
    assert not ran(build, "deck/vdmos_bulk_elsewhere")
    log = read_spice_text(rec.recorded(build, "deck/vdmos_bulk_elsewhere.log"))
    assert "requires the source and bulk to be the same node" in log
