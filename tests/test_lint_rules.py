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


def test_model_missing_reads_staged_include_closure(tmp_path: Path):
    models = tmp_path / "models with spaces.lib"
    models.write_text(".model DFAST D(Is=1e-12)\n")
    deck = '.include "models with spaces.lib"\nV1 in 0 1\nD1 in 0 DFAST\n.op\n.end\n'

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
