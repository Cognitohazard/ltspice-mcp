"""Seed lint rules, suppression, metadata, and version provenance."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ltspice_mcp.lib import lint_rules
from ltspice_mcp.lib.lint_rules import RULES, RULES_BY_ID, LintRule, lint_deck, linter_version


def _ids(
    text: str,
    tmp_path: Path,
    *,
    dialect: str | None = None,
    simulator: str = "LTspice",
    suppress=(),
) -> set[str]:
    findings = lint_deck(
        text,
        tmp_path / "deck.cir",
        dialect,
        simulator,
        suppress=suppress,
    )
    return {finding["rule_id"] for finding in findings}


_CLEAN = "V1 in 0 1\nR1 in out 1k\n.model DFAST D(Is=1e-12)\nD1 out 0 DFAST\n.op\n.end\n"


@pytest.mark.parametrize(
    ("rule_id", "deck", "dialect", "simulator"),
    [
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
            '.lib "models.lib" TT\n.op\n.end\n',
            "ngspice",
            "NGspiceSimulator",
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
            "directive-arity",
            "V1 in 0 1\nR1 out 1k\n.op\n.end\n",
            None,
            "LTspice",
        ),
        (
            "include-relative",
            '.include "models.lib"\n.op\n.end\n',
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
    ],
)
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


@pytest.mark.parametrize("rule_id", [rule.rule_id for rule in RULES])
def test_each_seed_rule_stays_quiet_on_clean_deck(
    rule_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(lint_rules, "current_ngbehavior", lambda: "hsa")

    assert rule_id not in _ids(_CLEAN, tmp_path)


def test_save_meas_coverage_accepts_saved_signal(tmp_path: Path):
    deck = "V1 in 0 1\n.save V(out)\n.meas tran peak MAX V(out)\n.tran 1u 1m\n.end\n"

    assert "save-meas-coverage" not in _ids(deck, tmp_path)


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
    never happened. The single-run path has always refused that deck; the
    experiment path has to say the same thing, and name the mechanism that
    does work here."""
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
    deck = '.include "models with spaces.lib"\nV1 in 0 1\nD1 in 0 DFAST\n.op\n.end\n'

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
    deck = f'.include "{previous}"\nD1 in 0 DDEEP\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


def test_cyclic_live_includes_terminate(tmp_path: Path):
    """Two live includes referencing each other must not hang the walk, and
    declarations found before the cycle closes still count."""
    first = tmp_path / "a.inc"
    second = tmp_path / "b.inc"
    first.write_text(f'.include "{second}"\n')
    second.write_text(f'.include "{first}"\n.model DCYC D(Is=1e-12)\n')
    deck = f'.include "{first}"\nD1 in 0 DCYC\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


def test_sectioned_lib_reference_resolves_through_the_walk(tmp_path: Path):
    """A ``.lib file section`` reference walks into the library file, and the
    section declarations inside it read as sections, not as missing files."""
    library = tmp_path / "corners.lib"
    library.write_text(".lib TT\n.model DTT D(Is=1e-12)\n.endl TT\n")
    deck = f'.lib "{library}" TT\nD1 in 0 DTT\nV1 in 0 1\n.op\n.end\n'

    assert "model-missing" not in _ids(deck, tmp_path)


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
