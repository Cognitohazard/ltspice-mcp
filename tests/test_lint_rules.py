"""Seed lint rules, suppression, metadata, and version provenance."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ltspice_mcp.lib import lint_rules
from ltspice_mcp.lib.lint_rules import RULES, RULES_BY_ID, LintRule, lint_deck, linter_version
from ltspice_mcp.lib.spice_validator import ARITY_CHECKS


def _ids(
    text: str,
    tmp_path: Path,
    *,
    dialect: str | None = None,
    simulator: str = "LTspice",
    suppress=(),
) -> set[str]:
    path = tmp_path / "deck.cir"
    # Read as an 8-bit file, the one kind that can hold a byte UTF-8 would not.
    findings = lint_deck(
        text, path, dialect, simulator, suppress=suppress, codecs={path: "cp1252"}
    )
    return {finding["rule_id"] for finding in findings}


_CLEAN = "V1 in 0 1\nR1 in out 1k\n.model DFAST D(Is=1e-12)\nD1 out 0 DFAST\n.op\n.end\n"


# Each seed rule, a deck that trips it, and the dialect and simulator it is
# live under. The clean-deck test lints in the same context, so a rule that
# only runs for one simulator is held quiet where it could actually fire.
_SEED_CASES = [
    (
        "save-meas-coverage",
        "V1 in 0 1\n.save V(in)\n.meas tran peak MAX V(out)\n.tran 1u 1m\n.end\n",
        None,
        "LTspice",
    ),
    (
        "meas-ngspice-batch",
        "V1 in 0 1\n.meas tran peak MAX V(in)\n.tran 1u 1m\n.end\n",
        "ngspice",
        "NGspiceSimulator",
    ),
    (
        "lib-section-ngspice",
        '* t\n.lib "models.lib" TT\n.op\n.end\n',
        "ngspice",
        "NGspiceSimulator",
    ),
    (
        "lib-section-ltspice",
        '* t\n.lib "models.lib" TT\n.op\n.end\n',
        None,
        "LTspice",
    ),
    (
        "analysis-count-ltspice",
        "V1 in 0 AC 1\nR1 in 0 1k\n.ac dec 10 1 1k\n.tran 1m\n.end\n",
        None,
        "LTspice",
    ),
    (
        "meas-function-ltspice",
        "V1 in 0 AC 1\nR1 in 0 1k\n.ac dec 10 1 1k\n.meas ac g FIND vdb(in) AT 100\n.end\n",
        None,
        "LTspice",
    ),
    (
        "byte-85-ltspice",
        "* t\nV1 a 0 1\nR1 a 0 1k\n* 1k to 10k\u2026R2 a 0 1k\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "node-control-byte-ltspice",
        "* t\nV1 n\u20acf 0 1\nR1 n\u20acf 0 1k\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "step-ngspice",
        "V1 in 0 1\nR1 in 0 {r}\n.param r=1k\n.step param r 1k 10k 1k\n.op\n.end\n",
        "ngspice",
        "NGspiceSimulator",
    ),
    (
        "model-missing",
        "V1 in 0 1\nD1 in 0 MISSING\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "element-arity",
        "V1 in 0 1\nR1 out 1k\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "bsource-value-prefix",
        "V1 in 0 1\nB1 out 0 {V(in)*2}\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "value-keyword-ltspice",
        "V1 in 0 1\nC1 in 0 C=1n\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "value-expression-remnant",
        "V1 in 0 1\nB1 out 0 V = V(in) + 1\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "include-relative",
        '* t\n.include "models.lib"\n.op\n.end\n',
        None,
        "LTspice",
    ),
    (
        "suffix-mega-milli",
        "V1 in 0 1\nR1 in 0 1M\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "temp-as-param",
        "V1 in 0 1\n.param TEMP=27\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "value-suffix-mojibake",
        "V1 in 0 1\nC1 in 0 23Âµ\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "value-suffix-nonascii",
        "V1 in 0 1\nR1 in 0 10Ω\n.op\n.end\n",
        None,
        "LTspice",
    ),
]
_CASE_IDS = [case[0] for case in _SEED_CASES]


def test_every_seed_rule_has_a_case():
    assert sorted(_CASE_IDS) == sorted(rule.rule_id for rule in RULES)


@pytest.mark.parametrize(("rule_id", "deck", "dialect", "simulator"), _SEED_CASES, ids=_CASE_IDS)
def test_each_seed_rule_fires(
    rule_id: str,
    deck: str,
    dialect: str | None,
    simulator: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(lint_rules, "current_ngbehavior", lambda: "kiltpsa")

    assert rule_id in _ids(
        deck,
        tmp_path,
        dialect=dialect,
        simulator=simulator,
    )


# Each rule runs on the clean deck under the dialect and simulator where it
# can fire, in place of its own trigger deck.
@pytest.mark.parametrize(
    ("rule_id", "dialect", "simulator"),
    [(rule_id, dialect, simulator) for rule_id, _deck, dialect, simulator in _SEED_CASES],
    ids=_CASE_IDS,
)
def test_each_seed_rule_stays_quiet_on_clean_deck(
    rule_id: str,
    dialect: str | None,
    simulator: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(lint_rules, "current_ngbehavior", lambda: "kiltpsa")

    assert rule_id not in _ids(_CLEAN, tmp_path, dialect=dialect, simulator=simulator)


def test_save_meas_coverage_accepts_saved_signal(tmp_path: Path):
    deck = "V1 in 0 1\n.save V(out)\n.meas tran peak MAX V(out)\n.tran 1u 1m\n.end\n"

    assert "save-meas-coverage" not in _ids(deck, tmp_path)


@pytest.mark.parametrize("mode", ["hsa", ""])
def test_explicit_launch_mode_overrides_session_lint_mode(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(lint_rules, "current_ngbehavior", lambda: "kiltpsa")
    text = '* t\n.lib "models.lib" tt\n.op\n.end\n'
    findings = lint_deck(
        text, tmp_path / "deck.cir", "ngspice", "NGspiceSimulator", ngbehavior=mode
    )
    assert "lib-section-ngspice" not in {row["rule_id"] for row in findings}
    assert "lib-section-ngspice" in _ids(text, tmp_path, dialect="ngspice")


def test_lib_section_ngspice_names_the_setting_that_fixes_it(tmp_path, monkeypatch):
    """The refusal is the only place a caller meets this: the deck never runs,
    so no simulator error follows it. A finding that says only what is wrong
    leaves the caller rewriting a correct PDK deck, so it has to name the
    setting that makes ngspice load the section."""
    monkeypatch.setattr(lint_rules, "current_ngbehavior", lambda: "kiltpsa")
    findings = lint_deck(
        '* t\n.lib "models.lib" tt\n.op\n.end\n',
        tmp_path / "deck.cir",
        "ngspice",
        "NGspiceSimulator",
    )
    finding = next(item for item in findings if item["rule_id"] == "lib-section-ngspice")

    reason = finding["evidence"]["reason"]
    assert '[simulator] ngbehavior = "hsa"' in reason
    assert "LTSPICE_MCP_NGBEHAVIOR=hsa" in reason
    assert "restart" in reason


def test_save_meas_coverage_distinguishes_voltage_and_current(tmp_path: Path):
    deck = "V1 out 0 1\n.save V(out)\n.meas tran peak MAX I(out)\n.tran 1u 1m\n.end\n"

    assert "save-meas-coverage" in _ids(deck, tmp_path)


def test_temp_as_param_finds_nonfirst_declaration(tmp_path: Path):
    deck = "V1 in 0 1\n.param bias=1 TEMP=27\n.op\n.end\n"

    assert "temp-as-param" in _ids(deck, tmp_path)


def test_temp_as_param_blocks_and_names_the_directives_that_work(tmp_path: Path):
    """A .param TEMP deck runs and returns one temperature's answers as many.

    Nothing downstream can detect that, so the finding has to stop submission
    rather than annotate it — and it has to say which directives do work.
    """
    findings = lint_deck(
        "V1 in 0 1\n.param TEMP=27\n.op\n.end\n",
        tmp_path / "deck.cir",
        None,
        "LTspice",
    )
    finding = next(item for item in findings if item["rule_id"] == "temp-as-param")

    assert RULES_BY_ID["temp-as-param"].disposition == "blocking"
    assert finding["severity"] == "error"
    reason = finding["evidence"]["reason"]
    for directive in (".temp", ".step temp", ".options temp"):
        assert directive in reason


def test_step_ngspice_says_what_to_use_instead(tmp_path: Path):
    """ngspice ignores a .step line in batch mode, so the deck runs once at the
    base value and reports no error — the sweep the caller asked for silently
    never happened. The lint has to say so, and name the mechanism that does
    work here."""
    findings = lint_deck(
        "V1 in 0 1\nR1 in 0 {r}\n.param r=1k\n.step param r 1k 10k 1k\n.op\n.end\n",
        tmp_path / "deck.cir",
        "ngspice",
        "NGspiceSimulator",
    )
    finding = next(item for item in findings if item["rule_id"] == "step-ngspice")

    assert finding["severity"] == "warning"
    assert "variations" in finding["evidence"]["reason"]


def test_step_ngspice_is_quiet_on_ltspice(tmp_path: Path):
    """LTspice runs .step natively; the warning is an ngspice fact only."""
    deck = "V1 in 0 1\nR1 in 0 {r}\n.param r=1k\n.step param r 1k 10k 1k\n.op\n.end\n"

    assert "step-ngspice" not in _ids(deck, tmp_path)


def test_model_missing_reads_staged_include_closure(tmp_path: Path):
    models = tmp_path / "models with spaces.lib"
    models.write_text(".model DFAST D(Is=1e-12)\n")
    deck = '* t\n.include "models with spaces.lib"\nV1 in 0 1\nD1 in 0 DFAST\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


def test_model_missing_resolves_through_staged_include_snapshots(tmp_path: Path):
    """A deck staged for a Windows simulator names its includes in Windows
    form, which nothing on the Linux side can re-read from disk; the staged
    include closure arrives as snapshots and must satisfy the model lookup,
    or a valid deck is refused as model-missing. The snapshot path is never
    written, so a lookup that read disk could not pass."""
    deck = '.include "C:\\Users\\u\\Temp\\staged\\amp.inc"\nX1 in out AMP\nV1 in 0 1\n.op\n.end\n'

    findings = lint_deck(
        deck,
        tmp_path / "deck.cir",
        None,
        "LTspice",
        includes=[(tmp_path / "staged" / "amp.inc", ".subckt AMP a b\nR1 a b 1k\n.ends AMP\n")],
    )

    assert "model-missing" not in {finding["rule_id"] for finding in findings}


def test_snapshot_serves_models_without_reading_staged_files(tmp_path: Path):
    """The snapshot is authoritative for staged content: the deck names its
    staged include by the exact path the snapshot declares, that path was
    never written to disk, and the model still resolves — so the staged
    lookup performed no disk read."""
    staged_include = tmp_path / "staged" / "models.inc"
    deck = f'.include "{staged_include}"\nD1 in 0 DFAST\nV1 in 0 1\n.op\n.end\n'

    findings = lint_deck(
        deck,
        tmp_path / "staged" / "deck.cir",
        None,
        "LTspice",
        includes=[(staged_include, ".model DFAST D(Is=1e-12)\n")],
    )

    assert "model-missing" not in {finding["rule_id"] for finding in findings}


def test_live_reference_nested_in_staged_include_is_still_read(tmp_path: Path):
    """The staged closure cannot carry a live (unstaged) include; the walk
    still follows one out of a staged file's snapshot to find its models."""
    live = tmp_path / "live.inc"
    live.write_text(".model DLIVE D(Is=1e-14)\n")
    staged_include = tmp_path / "staged" / "wrap.inc"
    deck = f'.include "{staged_include}"\nD1 in 0 DLIVE\nV1 in 0 1\n.op\n.end\n'

    findings = lint_deck(
        deck,
        tmp_path / "staged" / "deck.cir",
        None,
        "LTspice",
        includes=[(staged_include, f'.include "{live}"\n')],
    )

    assert "model-missing" not in {finding["rule_id"] for finding in findings}


def test_five_level_live_include_chain_resolves_models(tmp_path: Path):
    """Real PDK model trees sit about five includes down — the reason
    staging's depth budget is eight — so the live-include walk must reach as
    far as staging would stage, or an allowed chain lints as model-missing."""
    deep = tmp_path / "l5.inc"
    deep.write_text(".model DDEEP D(Is=1e-15)\n")
    previous = deep
    for level in (4, 3, 2, 1):
        link = tmp_path / f"l{level}.inc"
        link.write_text(f'.include "{previous}"\n')
        previous = link
    deck = f'* t\n.include "{previous}"\nD1 in 0 DDEEP\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


def test_cyclic_live_includes_terminate(tmp_path: Path):
    """Two live includes referencing each other must not hang the walk, and
    declarations found before the cycle closes still count."""
    first = tmp_path / "a.inc"
    second = tmp_path / "b.inc"
    first.write_text(f'.include "{second}"\n')
    second.write_text(f'.include "{first}"\n.model DCYC D(Is=1e-12)\n')
    deck = f'* t\n.include "{first}"\nD1 in 0 DCYC\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


def test_sectioned_lib_reference_resolves_through_the_walk(tmp_path: Path):
    """A ``.lib file section`` reference walks into the library file, and the
    section declarations inside it read as sections, not as missing files."""
    library = tmp_path / "corners.lib"
    library.write_text(".lib TT\n.model DTT D(Is=1e-12)\n.endl TT\n")
    deck = f'* t\n.lib "{library}" TT\nD1 in 0 DTT\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


@pytest.mark.parametrize(
    "deck",
    [
        # The ratioed pair of a bandgap or PTAT cell: an area factor after the
        # model, as a number or a parameter.
        "* t\n.model QN NPN\nV1 c 0 1\nQ1 c c 0 QN\nQ2 c c e QN 8\nR1 e 0 1k\n.op\n.end\n",
        "* t\n.model QN NPN\n.param N=8\nV1 c 0 1\nQ2 c c 0 QN {N}\n.op\n.end\n",
        "* t\n.model QN NPN\nV1 c 0 1\nQ1 c c 0 QN off\n.op\n.end\n",
        "* t\n.model QN NPN\nV1 c 0 1\nQ1 c c 0 sub QN 8\n.op\n.end\n",
        "* t\n.model JN NJF\nV1 d 0 1\nJ1 d 0 0 JN 2 off\n.op\n.end\n",
        "* t\n.model NCH NMOS\nV1 d 0 1\nM1 d d 0 0 NCH off\n.op\n.end\n",
        # ``params:`` introduces the overrides; the subckt name is before it.
        "* t\n.subckt mysub a b params: R=1k\nR1 a b {R}\n.ends mysub\n"
        "V1 n1 0 1\nX1 n1 0 mysub params: R=2k\n.op\n.end\n",
    ],
)
def test_model_missing_reads_the_model_past_trailing_tokens(tmp_path: Path, deck: str):
    assert "model-missing" not in _ids(deck, tmp_path)


@pytest.mark.parametrize(
    ("card", "model"),
    [
        ("Q1 c b e QX 8", "QX"),
        ("Q1 c b e QX off", "QX"),
        ("X1 n1 0 nosub params: R=2k", "nosub"),
    ],
)
def test_model_missing_still_names_an_undeclared_model(tmp_path: Path, card: str, model: str):
    findings = lint_deck(
        f"* t\nV1 c 0 1\n{card}\n.op\n.end\n", tmp_path / "deck.cir", None, "LTspice"
    )
    (finding,) = [item for item in findings if item["rule_id"] == "model-missing"]
    assert finding["evidence"]["model"] == model


@pytest.mark.parametrize(
    "title",
    ["Diode clamp test", "Bandgap reference", "Current mirror", "Mirror 1M load"],
)
def test_a_free_text_title_is_not_read_as_an_element(tmp_path: Path, title: str):
    """Line 1 of a netlist is its title, which both simulators skip. A title
    that starts with an element letter is still prose, not a card."""
    deck = f"{title}\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n"

    assert _ids(deck, tmp_path) == set()


@pytest.mark.parametrize(
    "card",
    [
        "M1 d g s s N1 L=1u W=1u IC=1,2,3",
        "Q1 c b e QN IC=0.7,5",
        "R1 a 0 1k tc=0.001,1e-6",
        "B1 a 0 R=V(a)*1k",
        "B1 a 0 P=1",
        "B1 c 0 V = V(a) + V(b)",
        "C1 a 0 Q=1n*x",
        "L1 b 0 Flux=1m*tanh(I(L1))",
        'V1 a 0 wavefile="in.wav" chan=0',
    ],
)
def test_valid_ltspice_cards_do_not_block(tmp_path: Path, card: str):
    deck = f"* t\n.model N1 NMOS\n.model QN NPN\n{card}\n.op\n.end\n"
    findings = lint_deck(deck, tmp_path / "deck.cir", None, "LTspice")

    assert [
        item for item in findings if RULES_BY_ID[item["rule_id"]].disposition == "blocking"
    ] == []


def test_a_spaced_expression_is_a_warning_naming_the_edit_limit(tmp_path: Path):
    """``V = V(a) + V(b)`` runs; what is certain is that a value edit would
    rewrite only its first span. The validator calls that a warning, and the
    linter reports it as one."""
    findings = lint_deck(
        "* t\nB1 c 0 V = V(a) + V(b)\n.op\n.end\n", tmp_path / "deck.cir", None, "LTspice"
    )
    (finding,) = findings

    assert finding["rule_id"] == "value-expression-remnant"
    assert finding["severity"] == "warning"


def test_each_arity_rule_reports_the_validators_severity(tmp_path: Path):
    """One card per arity check: each lints under its own rule id, at the
    severity its validator check declares."""
    deck = "* t\nR1 a 1k\nB1 b 0 {V(a)}\nC1 c d C=1n\nB2 e 0 V = V(a) + V(b)\n.op\n.end\n"
    findings = lint_deck(deck, tmp_path / "deck.cir", None, "LTspice")

    assert {item["rule_id"]: item["severity"] for item in findings} == ARITY_CHECKS


@pytest.mark.parametrize(
    "suppressed",
    ["bsource-value-prefix", "value-keyword-ltspice", "value-expression-remnant"],
)
def test_suppressing_another_arity_check_keeps_the_node_count(tmp_path: Path, suppressed: str):
    """Each arity check has its own rule id, so suppressing one to get past a
    finding the caller has judged does not also drop a one-node resistor."""
    deck = "* t\nV1 in 0 1\nR1 out 1k\n.op\n.end\n"

    assert "element-arity" in _ids(deck, tmp_path, suppress=[suppressed])


def test_b_source_resistor_form_blocks_on_ngspice(tmp_path: Path):
    """ngspice's B-source takes V= or I= only, so R= is a real refusal there."""
    deck = "* t\nV1 a 0 1\nB1 a 0 R=V(a)*1k\n.op\n.end\n"

    assert "bsource-value-prefix" in _ids(
        deck, tmp_path, dialect="ngspice", simulator="NGspiceSimulator"
    )


def test_meas_ngspice_batch_is_a_warning(tmp_path: Path):
    """ngspice runs the deck and skips only the .meas, and the run reports that
    skip itself; refusing the whole deck up front costs the caller the run."""
    findings = lint_deck(
        "V1 in 0 1\n.meas tran peak MAX V(in)\n.tran 1u 1m\n.end\n",
        tmp_path / "deck.cir",
        "ngspice",
        "NGspiceSimulator",
    )
    (finding,) = findings

    assert finding["rule_id"] == "meas-ngspice-batch"
    assert RULES_BY_ID["meas-ngspice-batch"].disposition == "warning"
    assert finding["severity"] == "warning"


class TestValueSuffixRule:
    """A non-ASCII character where a scale suffix goes.

    LTspice 24 and later write the micro sign as UTF-8 (bytes C2 B5). LTspice
    XVII decodes a deck as cp1252, sees 'Âµ', recognises neither character as
    a scale, and runs 23µ as 23 without a word: a factor of 1e6. That damage
    shows as a cp1252 reading of a UTF-8 lead byte ('Â', 'Î', 'Ã') and blocks.
    Any other symbol ('10Ω', '25°C') is also read as the bare number, which is
    usually what was meant, so it is a warning. A micro sign itself is not
    flagged: staging spells it 'u' before the deck is linted.
    """

    def _finding(self, deck: str, tmp_path: Path, rule_id: str, **kwargs) -> dict:
        findings = lint_deck(deck, tmp_path / "deck.cir", None, "LTspice", **kwargs)
        return next(item for item in findings if item["rule_id"] == rule_id)

    @pytest.mark.parametrize(
        ("token", "reads_as", "intended"),
        [("23Âµ", "23", "23u"), ("2.2Î¼F", "2.2", "2.2uF")],
    )
    def test_mis_decoded_micro_blocks_and_names_the_intended_value(
        self, tmp_path: Path, token: str, reads_as: str, intended: str
    ):
        finding = self._finding(
            f"* t\nC1 in 0 {token}\n.op\n.end\n", tmp_path, "value-suffix-mojibake"
        )

        assert RULES_BY_ID["value-suffix-mojibake"].disposition == "blocking"
        assert finding["severity"] == "error"
        assert finding["subject"] == token
        assert finding["at"] == {"file": str(tmp_path / "deck.cir"), "line": 2}
        assert finding["evidence"]["reads_as"] == reads_as
        assert finding["evidence"]["likely_intended"] == intended

    @pytest.mark.parametrize(
        "card",
        [
            # A micro sign encoded as UTF-8 twice, then read as cp1252.
            "C1 in 0 23" + "µ".encode().decode("cp1252").encode().decode("cp1252"),
            # Other characters damaged the same way: the lead byte is what is
            # seen, and the file's other values may have lost a micro sign.
            ".temp 25Â°C",
            "R1 in 0 10Î©",
        ],
    )
    def test_other_mis_decoded_characters_block(self, tmp_path: Path, card: str):
        finding = self._finding(
            f"* t\nV1 in 0 1\n{card}\n.op\n.end\n", tmp_path, "value-suffix-mojibake"
        )

        assert finding["severity"] == "error"
        assert "likely_intended" not in finding["evidence"]
        assert "cp1252" in finding["evidence"]["reason"]

    @pytest.mark.parametrize(
        ("card", "reads_as"),
        [
            ("R1 in 0 10Ω", "10"),
            (".temp 25°C", "25"),
            ("V2 s 0 SINE(0 1 1k 0 0 90°)", "90"),
        ],
    )
    def test_a_symbol_after_the_number_warns(self, tmp_path: Path, card: str, reads_as: str):
        """The simulator reads the bare number, which is what '10Ω' or '90°'
        means; nothing about the file says a scale was lost."""
        deck = f"* t\nV1 in 0 1\nR9 s 0 1k\n{card}\n.op\n.end\n"
        findings = lint_deck(deck, tmp_path / "deck.cir", None, "LTspice")
        (finding,) = [item for item in findings if item["rule_id"].startswith("value-suffix")]

        assert finding["rule_id"] == "value-suffix-nonascii"
        assert RULES_BY_ID["value-suffix-nonascii"].disposition == "warning"
        assert finding["severity"] == "warning"
        assert finding["evidence"]["reads_as"] == reads_as
        assert "likely_intended" not in finding["evidence"]

    @pytest.mark.parametrize(
        "deck",
        [
            "* t\nC1 in 0 23µ\n.param tau=10μ\n.op\n.end\n",
            "* 23Âµ in a comment\nC1 in 0 1n ; was 23Âµ\n.op\n.end\n",
            "C1 in 0 23Âµ is the title line\nR1 in 0 1k\n.op\n.end\n",
            "* t\nR1 nÂ1 N001Â 1k\n.op\n.end\n",
            "* t\nR1 in 0 10kΩ\n.op\n.end\n",
            '* t\nV1 in 0 PWL file="C:\\data\\10Âµs.txt"\n.op\n.end\n',
            "* t\n.include C:\\models\\10Âµ.lib\n.op\n.end\n",
            "* t\nV1 in 0 1\n.control\nlet x = 10Âµ\n.endc\n.end\n",
        ],
    )
    def test_quiet_where_the_character_is_not_a_scale_suffix(self, tmp_path: Path, deck: str):
        assert not {"value-suffix-mojibake", "value-suffix-nonascii"} & _ids(deck, tmp_path)

    def test_staged_include_is_scanned_and_named(self, tmp_path: Path):
        include = tmp_path / "staged" / "core.inc"
        finding = self._finding(
            f'* t\n.include "{include}"\nX1 in out core\n.op\n.end\n',
            tmp_path,
            "value-suffix-mojibake",
            includes=[(include, ".subckt core a b\nC1 a b 23Âµ\n.ends core\n")],
        )

        assert finding["at"] == {"file": str(include), "line": 2}

    def test_exporter_header_is_reported(self, tmp_path: Path):
        finding = self._finding(
            "* C:\\work\\rc.asc\n* Generated by LTspice 24.1.9 for Windows.\n"
            "C1 in 0 23Âµ\n.op\n.end\n",
            tmp_path,
            "value-suffix-mojibake",
        )

        assert finding["evidence"]["generated_by"] == "LTspice 24.1.9 for Windows"


def test_suppression_removes_named_rule(tmp_path: Path):
    deck = "V1 in 0 1\nR1 in 0 1M\n.op\n.end\n"

    assert "suffix-mega-milli" not in _ids(
        deck,
        tmp_path,
        suppress=["suffix-mega-milli"],
    )


_NGSPICE_CONTROL_DECK = (
    "* ngspice control-block deck\n"
    "V1 in 0 DC 1\n"
    "R1 in out 1k\n"
    "C1 out 0 1u\n"
    ".tran 1u 1m\n"
    ".control\n"
    "set filetype=ascii\n"
    "let vo = v(out)\n"
    "dc VDD 1.0 1.8 0.005\n"
    "meas dc vhalf find vo when v(in)=0.5\n"
    "foreach il 0 10m\n"
    "alter ILOAD = $il\n"
    "run\n"
    "end\n"
    "write out.raw\n"
    ".endc\n"
    ".end\n"
)


def test_control_block_contents_produce_no_findings(tmp_path: Path):
    # ngspice control commands share their first letter with SPICE element
    # prefixes (let->L, dc->D, meas->M, ...). The block is simulator script,
    # not netlist, so no rule may read inside it — otherwise the documented
    # ngspice .meas workaround gets its own deck refused.
    assert (
        _ids(
            _NGSPICE_CONTROL_DECK,
            tmp_path,
            dialect="ngspice",
            simulator="NGspiceSimulator",
        )
        == set()
    )


def test_registry_dispositions_are_explicit():
    assert RULES_BY_ID["save-meas-coverage"].disposition == "blocking"
    assert RULES_BY_ID["include-relative"].disposition == "warning"
    assert all(rule.disposition for rule in RULES)
    assert "op-degenerate" not in RULES_BY_ID


def test_rule_metadata_is_limited_to_fields_something_reads():
    """A rule carries only metadata a consumer acts on.

    ``disposition`` picks the finding's severity and drives the blocking gate;
    ``check`` is the rule body. Anything else every rule sets and nothing reads
    is worse than absent when it names an execution stage: ``lint_deck`` runs
    every rule unconditionally, so a rule declaring a later stage would run at
    preflight anyway. Add such a field together with the code that honours it.
    """
    assert {field.name for field in dataclasses.fields(LintRule)} == {
        "rule_id",
        "disposition",
        "check",
    }


def test_linter_version_is_stable_nonempty_string():
    assert isinstance(linter_version, str)
    assert linter_version


_BYTE_85_DECK = "* t\nV1 a 0 1\nR1 a 0 1k\n* 1k to 10k…R2 a 0 1k\n.op\n.end\n"


def test_an_ellipsis_in_a_utf8_deck_is_no_byte_85(tmp_path: Path):
    """UTF-8 spells an ellipsis E2 80 A6; only an 8-bit file holds byte 0x85."""
    path = tmp_path / "deck.cir"
    findings = lint_deck(_BYTE_85_DECK, path, None, "LTspice", codecs={path: "utf-8"})
    assert "byte-85-ltspice" not in {finding["rule_id"] for finding in findings}


def test_byte_85_is_one_line_to_a_known_xvii(tmp_path: Path):
    path = tmp_path / "deck.cir"
    findings = lint_deck(
        _BYTE_85_DECK, path, None, "LTspice", codecs={path: "cp1252"}, cp1252_reader="XVIIx64.exe"
    )
    assert "byte-85-ltspice" not in {finding["rule_id"] for finding in findings}


def test_byte_85_before_a_comment_or_the_line_end_changes_no_card(tmp_path: Path):
    deck = "* t\nV1 a 0 1\nR1 a 0 1k ; ten…\n* one…* two\n.op\n.end\n"
    assert "byte-85-ltspice" not in _ids(deck, tmp_path)


def test_byte_85_in_an_include_is_found_in_the_include(tmp_path: Path):
    deck = tmp_path / "deck.cir"
    include = tmp_path / "parts.lib"
    findings = lint_deck(
        '* t\n.inc "parts.lib"\nV1 a 0 1\n.op\n.end\n',
        deck,
        None,
        "LTspice",
        includes=[(include, "* parts…R9 a 0 1k\n")],
        codecs={deck: "utf-8", include: "cp1252"},
    )
    (finding,) = [f for f in findings if f["rule_id"] == "byte-85-ltspice"]
    assert finding["at"] == {"file": str(include), "line": 1}
    assert finding["evidence"]["read_as_cards"] == ["R9 a 0 1k"]


def test_a_node_name_in_a_utf8_deck_holds_no_control_byte(tmp_path: Path):
    """In UTF-8 a euro sign is three bytes LTspice reads as one character."""
    path = tmp_path / "deck.cir"
    deck = "* t\nV1 n\u20acf 0 1\nR1 n\u20acf 0 1k\n.op\n.end\n"
    findings = lint_deck(deck, path, None, "LTspice", codecs={path: "utf-8"})
    assert "node-control-byte-ltspice" not in {finding["rule_id"] for finding in findings}
