"""Seed lint rules, suppression, metadata, and version provenance."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib.lint_rules import RULES, RULES_BY_ID, LintRule, lint_deck, linter_version
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_lex_views import MeasCard
from ltspice_mcp.lib.spice_validator import (
    ARITY_CHECKS,
    validate_netlist_dangling_nodes,
    validate_netlist_directive_refs,
)
from tests.conftest import FIXTURES_DIR


def _ids(
    text: str,
    tmp_path: Path,
    *,
    dialect: str | None = None,
    simulator: str = "LTspice",
    suppress=(),
    codec: str = "cp1252",
    cp1252_reader: str | None = None,
) -> set[str]:
    # Read as an 8-bit file by default, the one kind that can hold a byte
    # UTF-8 would not.
    findings = lint_deck(
        text,
        tmp_path / "deck.cir",
        dialect,
        simulator,
        suppress=suppress,
        codec=codec,
        cp1252_reader=cp1252_reader,
    )
    return {finding["rule_id"] for finding in findings}


_CLEAN = "V1 in 0 1\nR1 in out 1k\n.model DFAST D(Is=1e-12)\nD1 out 0 DFAST\n.op\n.end\n"


# A comment holding byte 0x85, with a card after it on the line.
_BYTE_85_DECK = "* t\nV1 a 0 1\nR1 a 0 1k\n* 1k to 10k\u2026R2 a 0 1k\n.op\n.end\n"
# A node named with byte 0x80, a euro sign to cp1252.
_EURO_NODE_DECK = "* t\nV1 n\u20acf 0 1\nR1 n\u20acf 0 1k\n.op\n.end\n"

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
        "save-pattern-unrecorded",
        "V1 in 0 1\n.save V(i*)\n.meas tran peak MAX V(out)\n.tran 1u 1m\n.end\n",
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
        "meas-trig-degrees",
        "V1 in 0 SIN(0 1 1k)\n.meas tran c INTEG V(in)*cos(2*pi*1k*time)\n.tran 1m\n.end\n",
        None,
        "LTspice",
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
        "step-without-values-ltspice",
        "* t\n.param r=1k\nI1 0 a 1\nR1 a 0 {r}\n.step param r\n.op\n.end\n",
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
        "meas-name-ltspice",
        "V1 in 0 1\nR1 in 0 1k\n.tran 1m\n.meas tran k MAX V(in)\n.end\n",
        None,
        "LTspice",
    ),
    (
        "meas-unclosed-paren-ltspice",
        "V1 in 0 1\nR1 in 0 1k\n.tran 1m\n.meas tran x FIND V(in AT 0.5m\n.end\n",
        None,
        "LTspice",
    ),
    ("byte-85-ltspice", _BYTE_85_DECK, None, "LTspice"),
    ("node-control-byte-ltspice", _EURO_NODE_DECK, None, "LTspice"),
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
        "duplicate-param",
        "V1 d 0 1\nM1 d d 0 0 NMOS w=1u W=2u\n.model NMOS NMOS\n.op\n.end\n",
        None,
        "LTspice",
    ),
    (
        "unreadable-card",
        "V1 in 0 1\nR1 in 0 {1k\n.op\n.end\n",
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
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", "kiltpsa")

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
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", "kiltpsa")

    assert rule_id not in _ids(_CLEAN, tmp_path, dialect=dialect, simulator=simulator)


def test_save_meas_coverage_accepts_saved_signal(tmp_path: Path):
    deck = "V1 in 0 1\n.save V(out)\n.meas tran peak MAX V(out)\n.tran 1u 1m\n.end\n"

    assert "save-meas-coverage" not in _ids(deck, tmp_path)


#: Two measured nodes, out and vr, and a current, for the .save under test.
_PATTERN_DECK = (
    "* t\nV1 in 0 1\nR1 in out 1k\nR2 out vr 1k\nR3 vr 0 1k\n{save}\n"
    ".meas tran a MAX V(out)\n.meas tran b MAX V(vr)\n.meas tran c MAX I(R1)\n"
    ".tran 1u 1m\n.end\n"
)


def _coverage_findings(tmp_path: Path, save: str) -> dict[str, list[list[str]]]:
    """Each coverage rule's signals per finding, for the deck saved by ``save``."""
    found: dict[str, list[list[str]]] = {}
    for f in lint_deck(
        _PATTERN_DECK.format(save=save), tmp_path / "deck.cir", "ltspice", "LTspice"
    ):
        if f["rule_id"] == "save-meas-coverage":
            found.setdefault(f["rule_id"], []).append(f["evidence"]["missing_signals"])
        elif f["rule_id"] == "save-pattern-unrecorded":
            found.setdefault(f["rule_id"], []).append(f["evidence"]["unchecked_signals"])
    return found


@pytest.mark.parametrize(
    ("save", "expected"),
    [
        (
            ".save V(*)",
            {"save-pattern-unrecorded": [["v(out)"], ["v(vr)"], ["i(r1)"]]},
        ),
        (
            ".save V(*) I(R1)",
            {"save-pattern-unrecorded": [["v(out)"], ["v(vr)"]]},
        ),
        (
            ".save V(o*) V(st:rst) I(R1)",
            {"save-pattern-unrecorded": [["v(out)"], ["v(vr)"]]},
        ),
        (
            ".save I(*)",
            {"save-pattern-unrecorded": [["v(out)"], ["v(vr)"], ["i(r1)"]]},
        ),
        (".save V(out) V(vr) I(R1)", {}),
        (".save all", {}),
    ],
    ids=["v_pattern", "v_pattern_and_a_current", "prefix_pattern", "i_pattern", "named", "all"],
)
def test_a_save_pattern_is_not_read_as_a_name(
    tmp_path: Path, save: str, expected: dict[str, list[list[str]]]
):
    """Which signals a pattern saves, of either kind, is not recorded, so
    beside a pattern a signal the .save does not name is not refused as
    unsaved but said to be unchecked. A list of names alone is held as before."""
    assert _coverage_findings(tmp_path, save) == expected


def test_the_unchecked_signals_are_a_warning(tmp_path: Path):
    assert RULES_BY_ID["save-pattern-unrecorded"].disposition == "warning"
    findings = lint_deck(
        _PATTERN_DECK.format(save=".save V(*) I(R1)"), tmp_path / "deck.cir", "ltspice", "LTspice"
    )
    assert {f["severity"] for f in findings if f["rule_id"].startswith("save-")} == {"warning"}


@pytest.mark.parametrize("mode", ["hsa", ""])
def test_explicit_launch_mode_overrides_session_lint_mode(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", "kiltpsa")
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
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", "kiltpsa")
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


#: Probe spellings, with what each reader of a probe makes of them: the names
#: the save rule finds unsaved beside ``.save V(a) V(b) I(R1)``, and the names
#: the validator reads inside it. Both read a pair of two nodes as its nodes:
#: both LTspice builds measure ``V(a,b)`` from a saved ``V(a)`` and ``V(b)``.
_PROBE_READINGS = [
    pytest.param("V(a)", [], ["a"], id="a-node"),
    pytest.param("v( a )", [], ["a"], id="spaced"),
    pytest.param("V(a,b)", [], ["a", "b"], id="a-pair"),
    pytest.param("V(a, b)", [], ["a", "b"], id="a-spaced-pair"),
    pytest.param("I(R1)", [], ["R1"], id="a-current"),
    pytest.param("V(x1:out)", ["v(x1:out)"], ["x1:out"], id="a-node-in-a-subcircuit"),
]


@pytest.mark.parametrize(("probe", "unsaved", "names"), _PROBE_READINGS)
def test_each_reader_of_a_probe_reads_its_nodes(
    tmp_path: Path, probe: str, unsaved: list[str], names: list[str]
):
    deck = f"* t\nV1 in 0 1\nR1 in a 1k\nR2 in b 1k\n.meas tran m AVG {probe}\n.tran 1m\n.end\n"
    saved = deck.replace(".tran", ".save V(a) V(b) I(R1)\n.tran")
    (coverage,) = [
        f["evidence"]["missing_signals"]
        for f in lint_deck(saved, tmp_path / "deck.cir", "ltspice", "LTspice")
        if f["rule_id"] == "save-meas-coverage"
    ] or [[]]
    assert coverage == unsaved

    cards = lex_deck(deck).cards
    (meas,) = [card for card in cards if card.kind == "meas"]
    view_names = [name for probe in MeasCard.from_card(meas).signal_refs for name in probe.names]
    assert view_names == names
    dangling = {
        str(issue["message"]).split("'")[1] for issue in validate_netlist_dangling_nodes(cards)
    }
    assert dangling == {"a", "b"} - {name.casefold() for name in names}
    elsewhere = lex_deck(deck.replace("R1 in a 1k\nR2 in b 1k\n", "R9 in 0 1k\n")).cards
    undefined = [
        str(issue["message"]).split("'")[1] for issue in validate_netlist_directive_refs(elsewhere)
    ]
    assert undefined == [name for name in names if ":" not in name]


@pytest.mark.parametrize(
    ("save", "probe", "missing"),
    [
        pytest.param("V(a) V(b)", "V(a,b)", [], id="pair-from-both-nodes"),
        pytest.param("V(a)", "V(a,b)", ["v(b)"], id="pair-missing-a-node"),
        pytest.param("V(a,b)", "V(b)", [], id="saved-pair-saves-its-nodes"),
        pytest.param("V(a)", "V(a,0)", ["v(a,0)"], id="against-ground-as-written"),
        pytest.param("V(a)", "V(a,GND)", ["v(a,gnd)"], id="against-gnd-as-written"),
        pytest.param("V(a,0)", "V(a,0)", [], id="saved-as-written"),
    ],
)
def test_the_save_rule_reads_a_pair_of_two_nodes_as_its_nodes(
    tmp_path: Path, save: str, probe: str, missing: list[str]
):
    """Recorded on both builds: ``V(a,b)`` is measured from a saved ``V(a)``
    and ``V(b)``, and ``.save V(a,b)`` saves both nodes. A voltage against
    ground is not measured with only the node saved, so it stands as written."""
    deck = f"* t\nV1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.save {save}\n.tran 1m\n.meas tran m AVG {probe}\n.end\n"
    found = [
        f["evidence"]["missing_signals"]
        for f in lint_deck(deck, tmp_path / "deck.cir", "ltspice", "LTspice")
        if f["rule_id"] == "save-meas-coverage"
    ]
    assert found == ([missing] if missing else [])


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
        includes=[
            (tmp_path / "staged" / "amp.inc", ".subckt AMP a b\nR1 a b 1k\n.ends AMP\n", "utf-8")
        ],
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
        includes=[(staged_include, ".model DFAST D(Is=1e-12)\n", "utf-8")],
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
        includes=[(staged_include, f'.include "{live}"\n', "utf-8")],
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


class TestMeasTrigDegrees:
    """LTspice reads a trig function's angle in a .meas in degrees on its
    defaults, and in a B source in radians (recorded:
    test_recorded_ltspice_results.py::TestMeasurementAngleUnit)."""

    def test_each_measurement_names_the_angle_functions_it_calls(self, tmp_path: Path):
        deck = (
            "* phase of the fundamental\n"
            "V1 out 0 SIN(0 1 1k)\n"
            ".param fo=1k\n"
            ".meas tran a INTEG V(out)*cos(2*pi*fo*time)\n"
            ".meas tran b INTEG V(out)*SIN(2*pi*fo*time)\n"
            ".meas tran ph PARAM atan2(b,a)*180/pi + 0*cos(atan(1))\n"
            ".tran 2m\n"
            ".end\n"
        )
        findings = [
            f
            for f in lint_deck(deck, tmp_path / "deck.cir", None, "LTspice")
            if f["rule_id"] == "meas-trig-degrees"
        ]

        assert [(f["subject"], f["evidence"]["functions"]) for f in findings] == [
            ("a", ["cos"]),
            ("b", ["sin"]),
            ("ph", ["atan2", "cos", "atan"]),
        ]
        assert {f["severity"] for f in findings} == {"error"}
        reason = findings[0]["evidence"]["reason"]
        assert "RadianMeasure" in reason
        assert "B source" in reason

    @pytest.mark.parametrize(
        "card",
        [
            ".meas tran h PARAM sinh(1)+cosh(1)+tanh(1)+exp(1)+hypot(3,4)",
            ".meas tran cos FIND V(cos) AT 1m",
            ".meas tran x FIND V(sin_out) AT 1m",
            ".meas ac g FIND ph(V(out)) AT 1k",
        ],
    )
    def test_quiet_without_an_angle_function(self, tmp_path: Path, card: str):
        deck = f"* t\nV1 cos 0 1\nB1 sin_out 0 V=cos(2*pi*1k*time)\n{card}\n.tran 1m\n.end\n"

        assert "meas-trig-degrees" not in _ids(deck, tmp_path)

    def test_quiet_on_ngspice(self, tmp_path: Path):
        deck = "V1 in 0 1\n.meas tran p PARAM atan2(1,1)\n.tran 1m\n.end\n"

        assert "meas-trig-degrees" not in _ids(
            deck, tmp_path, dialect="ngspice", simulator="NGspiceSimulator"
        )


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
            includes=[(include, ".subckt core a b\nC1 a b 23Âµ\n.ends core\n", "utf-8")],
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
    ``check`` is the rule body; ``family`` is the simulator family a rule
    models, which ``lint_deck`` runs it for alone and the capability table
    reads. Anything else every rule sets and nothing reads is worse than
    absent when it names an execution stage: ``lint_deck`` runs every rule of
    the deck's family unconditionally, so a rule declaring a later stage would
    run at preflight anyway. Add such a field together with the code that
    honours it.
    """
    assert {field.name for field in dataclasses.fields(LintRule)} == {
        "rule_id",
        "disposition",
        "check",
        "family",
    }


def test_linter_version_is_stable_nonempty_string():
    assert isinstance(linter_version, str)
    assert linter_version


def test_an_ellipsis_in_a_utf8_deck_is_no_byte_85(tmp_path: Path):
    """UTF-8 spells an ellipsis E2 80 A6; only an 8-bit file holds byte 0x85."""
    assert "byte-85-ltspice" not in _ids(_BYTE_85_DECK, tmp_path, codec="utf-8")


def test_byte_85_is_one_line_to_a_known_xvii(tmp_path: Path):
    assert "byte-85-ltspice" not in _ids(_BYTE_85_DECK, tmp_path, cp1252_reader="XVIIx64.exe")


def test_byte_85_before_a_comment_or_the_line_end_changes_no_card(tmp_path: Path):
    deck = "* t\nV1 a 0 1\nR1 a 0 1k ; ten\u2026\n* one\u2026* two\n.op\n.end\n"
    assert "byte-85-ltspice" not in _ids(deck, tmp_path)


def test_byte_85_in_an_include_is_found_in_the_include(tmp_path: Path):
    include = tmp_path / "parts.lib"
    findings = lint_deck(
        '* t\n.inc "parts.lib"\nV1 a 0 1\n.op\n.end\n',
        tmp_path / "deck.cir",
        None,
        "LTspice",
        includes=[(include, "* parts\u2026R9 a 0 1k\n", "cp1252")],
    )
    (finding,) = [f for f in findings if f["rule_id"] == "byte-85-ltspice"]
    assert finding["at"] == {"file": str(include), "line": 1}
    assert finding["evidence"]["read_as_cards"] == ["R9 a 0 1k"]


def test_a_node_name_in_a_utf8_deck_holds_no_control_byte(tmp_path: Path):
    """In UTF-8 a euro sign is three bytes LTspice reads as one character."""
    assert "node-control-byte-ltspice" not in _ids(_EURO_NODE_DECK, tmp_path, codec="utf-8")


# A resistor with an unclosed brace and a .model with an unclosed quote: no
# rule can split either card into its parts.
_UNREADABLE = (
    "* t\nV1 in 0 5\nR1 in out {1k\nC1 out 0 1n\n"
    '.model D1 D(IS="1e-14\n.meas op vr FIND I(R1)\n.op\n.end\n'
)


def test_a_card_no_rule_can_read_is_reported_where_it_is(tmp_path):
    findings = lint_deck(_UNREADABLE, tmp_path / "deck.cir", "ltspice", "LTspice")

    unread = [f for f in findings if f["rule_id"] == "unreadable-card"]
    assert [f["at"]["line"] for f in unread] == [3, 5]
    part, model = (f["evidence"]["detail"] for f in unread)
    assert "the checks could not split this card" in part
    assert "Check for a missing } matching the {." in part
    # Only a card that may connect nodes has its words counted as connected.
    assert "count as connected and defined" in part
    assert "count as connected" not in model
    assert RULES_BY_ID["unreadable-card"].disposition == "warning"


def test_an_include_that_cannot_be_read_in_a_live_file_leaves_models_unknown(tmp_path):
    """A live include holding an include card it cannot read: which file that
    card names, and so which models it declares, is not known. The scanner's
    error escaped lint_deck; now no model is reported missing on the guess,
    and the reply says so at that card."""
    (tmp_path / "live.inc").write_text('.include "models.lib\n')
    deck = '* t\n.include "live.inc"\nV1 a 0 1\nD1 a 0 mydiode\n.op\n.end\n'

    findings = lint_deck(deck, tmp_path / "deck.cir", "ltspice", "LTspice")

    assert "model-missing" not in {f["rule_id"] for f in findings}
    (unread,) = [f for f in findings if f["rule_id"] == "unreadable-card"]
    assert unread["at"] == {"file": str(tmp_path / "live.inc"), "line": 1}
    assert "model-missing reported nothing" in unread["evidence"]["detail"]
    readable = deck.replace("live.inc", "other.inc")
    (tmp_path / "other.inc").write_text("* nothing here\n")
    assert "model-missing" in _ids(readable, tmp_path)


def test_a_card_that_cannot_be_read_in_a_staged_include_is_reported_there(tmp_path):
    staged = tmp_path / "staged" / "parts.inc"
    deck = '* t\n.include "parts.inc"\nV1 a 0 1\n.op\n.end\n'

    findings = lint_deck(
        deck,
        tmp_path / "deck.cir",
        "ltspice",
        "LTspice",
        includes=[(staged, "R9 a 0 1k\nR1 a 0 {1k\n", "utf-8")],
    )

    (unread,) = [f for f in findings if f["rule_id"] == "unreadable-card"]
    assert unread["at"] == {"file": str(staged), "line": 2}


def test_a_measurement_with_a_parenthesis_left_open_blocks_an_ltspice_run(tmp_path):
    """LTspice 26 refuses the recorded deck at that .meas and runs nothing;
    XVII runs it without the measurement (log/meas_bad_syntax). What ngspice
    does with it is not recorded, so there it is the warning any card no rule
    can read is."""
    source = FIXTURES_DIR / "ltspice_recorded" / "inputs" / "log" / "meas_bad_syntax.cir"
    deck = source.read_text(encoding="utf-8")

    def found(family: str) -> list[tuple[str, int]]:
        findings = lint_deck(deck, tmp_path / "deck.cir", family, None)
        return [
            (f["rule_id"], f["at"]["line"])
            for f in findings
            if f["rule_id"] in ("meas-unclosed-paren-ltspice", "unreadable-card")
        ]

    assert found("ltspice") == [("meas-unclosed-paren-ltspice", 7)]
    assert found("ngspice") == [("unreadable-card", 7)]
    assert RULES_BY_ID["meas-unclosed-paren-ltspice"].disposition == "blocking"


# A directive with a bracket or quote left open: each was read for its name or
# parts by some rule, which raised the lexer's error out of lint_deck.
_OPEN_DIRECTIVES = [
    ".tran 0 1m )",
    ".save V(in",
    ".param r={1k",
    ".step param r list (1k 2k",
    ".options plotwinsize=(0",
]


@pytest.mark.parametrize("family", ["ltspice", "ngspice"])
@pytest.mark.parametrize("directive", _OPEN_DIRECTIVES)
def test_a_directive_no_rule_can_read_is_reported_not_raised(tmp_path, family, directive):
    deck = f"* t\nV1 in 0 1\nR1 in 0 {{r}}\n.param r=1k\n{directive}\n.op\n.end\n"

    findings = lint_deck(deck, tmp_path / "deck.cir", family, None)

    unread = [f for f in findings if f["rule_id"] == "unreadable-card"]
    assert [(f["at"]["line"], f["subject"]) for f in unread] == [(5, directive)]


def test_a_save_no_rule_can_read_may_save_what_a_measurement_asks_for(tmp_path):
    # '.save V(in' may save V(in); the other .save does not, so reading only it
    # would refuse the measurement on a guess.
    deck = (
        "* t\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.save V(out)\n.save V(in\n"
        ".meas tran vin MAX V(in)\n.tran 1m\n.end\n"
    )

    assert "save-meas-coverage" not in _ids(deck, tmp_path)
    assert "save-meas-coverage" in _ids(deck.replace(".save V(in\n", ""), tmp_path)
