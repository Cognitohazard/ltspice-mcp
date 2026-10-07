"""How the server reads a deck, held against what LTspice did with the same deck.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: decks
LTspice 26 and LTspice XVII each ran. Most are operating points of a 1 A
current source into one resistor, so the node voltage in the recorded raw is
the resistance LTspice read off the card. The server's value parser, lexer,
lint and arity rules read the same card and have to reach the same answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.encoding import decode_spice_bytes_with_encoding, read_spice_text
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.hierarchy_expr import evaluate
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build, is_cp1252_ltspice_executable
from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import read_instance
from ltspice_mcp.lib.spice_validator import (
    drop_title_card,
    validate_directive,
    validate_netlist_arity,
)
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS
from tests.test_inspect_tools import _run

MICRO = "µ"


def deck_text(case_id: str) -> str:
    return read_spice_text(INPUTS / rec.CASES.case(case_id).source)


def resistances(case_id: str) -> dict[str, str]:
    """Each probe node of a suffix deck and the value written on its resistor."""
    values: dict[str, str] = {}
    for card in drop_title_card(lex(deck_text(case_id)).cards):
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

#: Values LTspice reads and the server's parser refuses, with what LTspice
#: makes of each (the same on both recorded builds). Two families: a letter
#: that is no scale suffix, which LTspice skips, and the "3k4" shorthand that
#: puts the suffix where the decimal point goes, which both builds accept as
#: installed. ``parse_spice_value`` raises on all of them; a caller that
#: validates a value with it turns away a deck LTspice would run.
READ_BY_LTSPICE_ONLY = {
    "2Hz": 2.0,
    "3V": 3.0,
    "2ohm": 2.0,
    "2A": 2.0,
    "7x": 7.0,
    "5s": 5.0,
    "2Ohms": 2.0,
    "4H": 4.0,
    "6W": 6.0,
    "9V1": 9.0,
    "1k5": 1500.0,
    "4R7": 4.7,
    "2M2": 2.2e-3,
    "3u3": 3.3e-6,
    "1Meg5": 1.5e6,
}


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUFFIX_DECKS)))
def test_a_value_the_server_parses_is_the_value_ltspice_read(
    build: str, case_id: str, tmp_path: Path
):
    """Scale suffixes in either case, ``M`` against ``Meg``, ``mil``, unit
    tails, and words that merely begin with a suffix (``1MHz`` is a millihertz
    to LTspice, ``1milli`` a mil)."""
    voltages = rec.operating_point(build, case_id, tmp_path)
    written = resistances(case_id)
    assert len(written) >= 5
    refused: dict[str, float] = {}
    for node, spelling in written.items():
        read = voltages[f"v({node})"]
        try:
            parsed = parse_spice_value(spelling)
        except ValueError:
            refused[spelling] = read
            continue
        # A raw stores a node voltage in four bytes.
        assert parsed == pytest.approx(read, rel=1e-6), spelling
    assert refused == pytest.approx(
        {spelling: READ_BY_LTSPICE_ONLY[spelling] for spelling in refused}, rel=1e-6
    )


def test_every_value_only_ltspice_reads_is_recorded():
    """The table above is what the recordings hold, no more and no less."""
    spellings = {
        spelling
        for case_id in SUFFIX_DECKS
        for spelling in resistances(case_id).values()
        if spelling in READ_BY_LTSPICE_ONLY
    }
    assert spellings == set(READ_BY_LTSPICE_ONLY)


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
    path = INPUTS / rec.CASES.case(case_id).source
    return {
        token
        for finding in lint_deck(deck_text(case_id), path, "ltspice", "LTspice")
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

    def test_the_first_line_is_a_title_even_when_it_reads_as_a_card(
        self, build: str, tmp_path: Path
    ):
        # Line 1 is "I1 0 a 5"; read as a card it would duplicate I1.
        assert self.voltage(build, "deck/title_card", tmp_path) == pytest.approx(1000.0)
        kept = drop_title_card(self.cards("deck/title_card"))
        assert [card.body for card in kept if card.name == "I1"] == ["I1 0 a 1"]

    def test_the_first_line_is_a_title_even_when_it_reads_as_a_directive(self, build: str):
        """Line 1 is ``.param r=2k``. LTspice does not define ``r`` from it, so
        the resistor that uses ``{r}`` fails the run; the server's checks must
        not count the line as a directive either."""
        assert not ran(build, "deck/title_directive")
        kept = drop_title_card(self.cards("deck/title_directive"))
        assert not [card for card in kept if card.kind == "param"]

    def test_cards_after_end_are_not_read(self, build: str, tmp_path: Path):
        # A second resistor after .end would halve the voltage.
        assert self.voltage(build, "deck/after_end", tmp_path) == pytest.approx(1000.0)
        (after,) = [c for c in self.cards("deck/after_end") if c.name == "R2"]
        assert after.trailing

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
            assert (ground["scope"], ground["name"]) == ([], "0"), instance
        port = rows[("X2", "R1")]["nodes"][1]
        assert (port["scope"], port["name"]) == ([], "mid")


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
}

FORMS = sorted(rec.cases_of("deck-forms"))


def refusals(case_id: str) -> set[str]:
    path = INPUTS / rec.CASES.case(case_id).source
    text = deck_text(case_id)
    lint = {
        f["rule_id"]
        for f in lint_deck(text, path, "ltspice", "LTspice")
        if f["severity"] == "error"
    }
    arity = {
        str(issue["check"])
        for issue in validate_netlist_arity(drop_title_card(lex(text).cards), simulator="LTspice")
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
    if case_id in REFUSED or case_id in NOT_REFUSED_YET:
        pytest.skip("not a form the checks accept")
    assert ran(build, case_id)


#: Decks LTspice refuses that no check turns away before the run.
NOT_REFUSED_YET = {
    # "More than one analysis specified." The one-analysis rule is written
    # down in spice_validator.EXCLUSIVE_ANALYSIS_KINDS and enforced nowhere.
    "deck/ac_and_tran",
    # vdb(), phase() and group_delay() in a .meas: caught by the directive
    # check below, which is not part of the lint.
    "deck/meas_function_vdb",
    "deck/meas_function_phase",
    "deck/meas_function_group_delay",
}


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
# What a deck means
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
class TestDeckSemantics:
    def test_a_parameter_named_temp_never_sets_the_temperature(self, build: str, tmp_path: Path):
        """LTspice 26 refuses ``.param temp=50``; XVII runs the deck at 27
        degrees as if the line were not there. The lint refuses it for both."""
        path = INPUTS / "deck/param_temp.cir"
        findings = lint_deck(deck_text("deck/param_temp"), path, "ltspice", "LTspice")
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
            if fact["value"] is not None:
                assert fact["value"] == pytest.approx(read[f"v({node})"]), reference
        # The two the server cannot be wrong about because it declines them.
        assert rows[("R1",)]["value"]["value"] is None
        assert rows[("R4",)]["value"]["value"] is None
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
    sections: it looks for a file called ``corners.lib tt`` and stops. Nothing
    in the lint says so before the run."""
    assert not ran(build, "deck/lib_section")
    log = read_spice_text(rec.recorded(build, "deck/lib_section.log"))
    if rec.generation(build) == "xvii":
        assert 'Could not open library file "corners.lib tt"' in log
    else:
        assert "File not found." in log
        assert ".lib corners.lib tt" in log
    path = INPUTS / "deck/lib_section.cir"
    findings = lint_deck(deck_text("deck/lib_section"), path, "ltspice", "LTspice")
    assert [f["rule_id"] for f in findings if f["severity"] == "error"] == []
