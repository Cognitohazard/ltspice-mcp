"""Deterministic preflight lint registry for experiment decks."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.deck_staging import (
    is_absolute_reference,
    scan_include_references,
)
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.simulator import current_ngbehavior
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import InstanceLine
from ltspice_mcp.lib.spice_validator import PROBE_REF_RE, validate_netlist_arity

Disposition = Literal["blocking", "warning", "observation"]
Phase = Literal["preflight", "staging", "postrun"]
LintFinding = dict[str, Any]

linter_version = "1"

_SIGNAL_RE = PROBE_REF_RE
_MILLI_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)M(?![A-Za-z])"
)
_MODEL_PREFIXES = frozenset({"D", "M", "Q", "J", "X", "S", "W"})

RuleCheck = Callable[["_LintContext", "LintRule"], list[LintFinding]]


@dataclass(frozen=True)
class LintRule:
    """Metadata and implementation for one deterministic lint rule."""

    rule_id: str
    disposition: Disposition
    phase: Phase
    provenance: str
    check: RuleCheck


@dataclass(frozen=True)
class _LintContext:
    text: str
    path: Path
    cards: list[SpiceCard]
    dialect: str | None
    simulator_name: str

    @property
    def ngspice(self) -> bool:
        return self.dialect == "ngspice" or "ngspice" in self.simulator_name.casefold()


def _finding(
    context: _LintContext,
    rule: LintRule,
    *,
    line: int,
    subject: str,
    evidence: Any,
) -> LintFinding:
    severity = {
        "blocking": "error",
        "warning": "warning",
        "observation": "observation",
    }[rule.disposition]
    return {
        "rule_id": rule.rule_id,
        "severity": severity,
        "ok": False,
        "evidence": evidence,
        "at": {"file": str(context.path), "line": line},
        "subject": subject,
    }


def _directive_head(card: SpiceCard) -> str:
    tokens = tokenize_body(card.body)
    return tokens[0].text.casefold() if tokens else ""


def _save_meas_coverage(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    saves = [
        card
        for card in context.cards
        if card.kind == "directive" and _directive_head(card) == ".save"
    ]
    measurements = [card for card in context.cards if card.kind == "meas"]
    if not saves or not measurements:
        return []
    saved_text = " ".join(card.body for card in saves)
    if re.search(r"(?i)(?:^|\s)\.save\s+(?:all\b|\*)", saved_text):
        return []
    saved = {_normalize_signal(match.group(0)) for match in _SIGNAL_RE.finditer(saved_text)}
    findings = []
    for card in measurements:
        required = {_normalize_signal(match.group(0)) for match in _SIGNAL_RE.finditer(card.body)}
        missing = sorted(required - saved)
        if missing:
            findings.append(
                _finding(
                    context,
                    rule,
                    line=card.line_start,
                    subject=card.name or ".meas",
                    evidence={
                        "missing_signals": missing,
                        "saved_signals": sorted(saved),
                        "directive": card.body,
                    },
                )
            )
    return findings


def _meas_ngspice_batch(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    if not context.ngspice:
        return []
    return [
        _finding(
            context,
            rule,
            line=card.line_start,
            subject=card.name or ".meas",
            evidence={
                "directive": card.body,
                "reason": "ngspice batch mode with a raw output skips top-level .meas",
            },
        )
        for card in context.cards
        if card.kind == "meas" and card.scope == ()
    ]


def _lib_section_ngspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    if not context.ngspice:
        return []
    mode = (current_ngbehavior() or "").casefold()
    if "lt" not in mode and "ps" not in mode:
        return []
    findings = []
    for card in context.cards:
        if card.kind != "directive":
            continue
        tokens = tokenize_body(card.body)
        if len(tokens) < 3 or tokens[0].text.casefold() != ".lib":
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=tokens[2].text.strip("\"'"),
                evidence={
                    "directive": card.body,
                    "ngbehavior": mode,
                    "reason": (
                        "this compatibility mode treats a sectioned .lib as plain includes"
                    ),
                },
            )
        )
    return findings


def _model_missing(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    declared = _declared_models(context.cards)
    declared.update(_models_from_staged_dependencies(context))
    findings = []
    for card in context.cards:
        if card.kind != "instance" or not card.name:
            continue
        try:
            instance = InstanceLine.from_card(card)
        except ValueError:
            continue
        if instance.ref[:1].upper() not in _MODEL_PREFIXES or not instance.model:
            continue
        if instance.model.casefold() in declared:
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=instance.ref,
                evidence={
                    "model": instance.model,
                    "directive": card.body,
                    "declared_models": sorted(declared),
                },
            )
        )
    return findings


def _declared_models(cards: list[SpiceCard]) -> set[str]:
    return {
        card.name.casefold() for card in cards if card.kind in {"model", "subckt"} and card.name
    }


def _models_from_staged_dependencies(context: _LintContext) -> set[str]:
    """Collect declarations from the staged include closure, to depth three."""
    declared: set[str] = set()
    visited: set[Path] = set()

    def walk(cards: list[SpiceCard], source: Path, depth: int) -> None:
        if depth >= 3:
            return
        for reference in scan_include_references(cards, source):
            raw = reference.raw_path.replace("\\", "/")
            dependency = Path(raw)
            dependency = dependency if dependency.is_absolute() else source.parent / dependency
            try:
                resolved = dependency.resolve(strict=True)
            except OSError:
                continue
            if resolved in visited or not resolved.is_file():
                continue
            visited.add(resolved)
            try:
                nested = lex(read_spice_text(resolved)).cards
            except (OSError, UnicodeError, ValueError):
                continue
            declared.update(_declared_models(nested))
            walk(nested, resolved, depth + 1)

    walk(context.cards, context.path, 0)
    return declared


def _directive_arity(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    simulator = "ngspice" if context.ngspice else "LTspice"
    return [
        _finding(
            context,
            rule,
            line=raw_line if isinstance(raw_line := issue.get("line", 1), int) else 1,
            subject=str(issue.get("directive", "instance")),
            evidence={
                "message": issue.get("message"),
                "suggestion": issue.get("suggestion"),
            },
        )
        for issue in validate_netlist_arity(context.cards, simulator=simulator)
    ]


def _include_relative(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    findings = []
    for reference in scan_include_references(context.cards, context.path):
        if is_absolute_reference(reference.raw_path):
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=reference.card.line_start,
                subject=reference.raw_path,
                evidence={
                    "directive": reference.card.body,
                    "reason": "relative references depend on deck placement",
                },
            )
        )
    return findings


def _suffix_mega_milli(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    findings = []
    for card in context.cards:
        if card.kind in {"comment", "blank"}:
            continue
        matches = [match.group(0) for match in _MILLI_SUFFIX_RE.finditer(card.body)]
        if not matches:
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=matches[0],
                evidence={
                    "tokens": matches,
                    "reason": "SPICE interprets M as milli; mega is written Meg",
                },
            )
        )
    return findings


def _temp_as_param(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    findings = []
    for card in context.cards:
        if card.kind != "param":
            continue
        names = [
            token.key
            for token in tokenize_body(card.body)[1:]
            if token.kind == TokenKind.KEY_VALUE and token.key
        ]
        if not any(name.casefold() == "temp" for name in names):
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject="TEMP",
                evidence={
                    "directive": card.body,
                    "reason": "temperature is a simulator axis; use .temp or .step temp",
                },
            )
        )
    return findings


def _normalize_signal(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


RULES: tuple[LintRule, ...] = (
    LintRule(
        "save-meas-coverage",
        "blocking",
        "preflight",
        "SPICE save-list and measurement dependency semantics",
        _save_meas_coverage,
    ),
    LintRule(
        "meas-ngspice-batch",
        "blocking",
        "preflight",
        "ngspice batch-mode diagnostic",
        _meas_ngspice_batch,
    ),
    LintRule(
        "lib-section-ngspice",
        "blocking",
        "preflight",
        "ngspice compatibility-mode behavior",
        _lib_section_ngspice,
    ),
    LintRule(
        "model-missing",
        "blocking",
        "staging",
        "local model/subcircuit declarations",
        _model_missing,
    ),
    LintRule(
        "directive-arity",
        "blocking",
        "preflight",
        "shared SPICE arity validator",
        _directive_arity,
    ),
    LintRule(
        "include-relative",
        "warning",
        "preflight",
        "SPICE include resolution semantics",
        _include_relative,
    ),
    LintRule(
        "suffix-mega-milli",
        "warning",
        "preflight",
        "SPICE engineering-suffix semantics",
        _suffix_mega_milli,
    ),
    LintRule(
        "temp-as-param",
        "warning",
        "preflight",
        "SPICE temperature-axis semantics",
        _temp_as_param,
    ),
)

RULES_BY_ID: dict[str, LintRule] = {rule.rule_id: rule for rule in RULES}


def lint_deck(
    deck_text: str,
    path: Path,
    dialect: str | None,
    simulator: type | str | None,
    *,
    suppress: list[str] | set[str] | tuple[str, ...] = (),
) -> list[LintFinding]:
    """Run all unsuppressed rules and return fixable findings only."""
    suppressed = set(suppress)
    simulator_name = (
        simulator
        if isinstance(simulator, str)
        else simulator.__name__
        if simulator is not None
        else ""
    )
    context = _LintContext(
        text=deck_text,
        path=path,
        cards=lex(deck_text).cards,
        dialect=dialect,
        simulator_name=simulator_name,
    )
    findings: list[LintFinding] = []
    for rule in RULES:
        if rule.rule_id in suppressed:
            continue
        findings.extend(rule.check(context, rule))
    return findings
