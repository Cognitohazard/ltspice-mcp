"""Seed lint rules, suppression, metadata, and version provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib import lint_rules
from ltspice_mcp.lib.lint_rules import RULES, RULES_BY_ID, lint_deck, linter_version


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


def test_registry_dispositions_and_phases_are_explicit():
    assert RULES_BY_ID["save-meas-coverage"].disposition == "blocking"
    assert RULES_BY_ID["include-relative"].disposition == "warning"
    assert RULES_BY_ID["model-missing"].phase == "staging"
    assert all(rule.provenance for rule in RULES)
    assert "op-degenerate" not in RULES_BY_ID


def test_linter_version_is_stable_nonempty_string():
    assert isinstance(linter_version, str)
    assert linter_version
