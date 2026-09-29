"""Deterministic preflight lint registry for experiment decks."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.deck_staging import (
    DEFAULT_INCLUDE_DEPTH,
    is_absolute_reference,
    resolve_reference,
    scan_include_references,
)
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.simulator import current_ngbehavior
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_ops import ValueSuffixSite, value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import InstanceLine
from ltspice_mcp.lib.spice_validator import (
    PROBE_REF_RE,
    drop_title_card,
    validate_netlist_arity,
)

Disposition = Literal["blocking", "warning", "observation"]
LintFinding = dict[str, Any]

linter_version = "3"

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
    check: RuleCheck


@dataclass(frozen=True)
class _LintContext:
    text: str
    path: Path
    cards: list[SpiceCard]
    dialect: str | None
    simulator_name: str
    # The staged include closure, as (staged path, staged text) snapshots. On
    # a deck staged for a Windows simulator the rewritten references cannot be
    # re-read from the Linux side, so the snapshots are the authoritative
    # source for declarations the deck reaches through an include.
    includes: tuple[tuple[Path, str], ...] = ()
    ngbehavior: str | None = None

    @property
    def ngspice(self) -> bool:
        return self.dialect == "ngspice" or "ngspice" in self.simulator_name.casefold()

    @functools.cached_property
    def value_suffix_sites(self) -> tuple[tuple[Path, str | None, ValueSuffixSite], ...]:
        """Non-ASCII suffix sites in the deck and its staged includes.

        Each comes with the file it is in and the writer that file's header
        names. The deck's title line is dropped; an include has none. A file
        that is all ASCII has no site and is not lexed.
        """
        found = [
            (self.path, deck_generator(self.text), site)
            for site in value_suffix_sites(drop_title_card(self.cards))
        ]
        for path, text in self.includes:
            if not text.isascii():
                generator = deck_generator(text)
                found.extend(
                    (path, generator, site) for site in value_suffix_sites(lex(text).cards)
                )
        return tuple(found)


def _finding(
    context: _LintContext,
    rule: LintRule,
    *,
    line: int,
    subject: str,
    evidence: Any,
    file: Path | None = None,
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
        "at": {"file": str(file or context.path), "line": line},
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


def _step_ngspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    if not context.ngspice:
        return []
    findings = []
    for card in context.cards:
        if card.kind != "directive" or _directive_head(card) != ".step":
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=".step",
                evidence={
                    "directive": card.body,
                    "reason": (
                        "ngspice has no .step: in batch mode it ignores the line, so "
                        "the deck runs once at the base value and reports no error — "
                        "the sweep never happens. Run the sweep as run_experiments "
                        "variations (one deck per value) instead, or set the parameter "
                        "to a fixed value."
                    ),
                },
            )
        )
    return findings


def _lib_section_ngspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    if not context.ngspice:
        return []
    mode = (
        context.ngbehavior if context.ngbehavior is not None else current_ngbehavior() or ""
    ).casefold()
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
    """Collect declarations the deck reaches through its include references.

    The staged snapshots in ``context.includes`` are authoritative for the
    staged closure: each is lexed once, in memory, and its staged path is
    seeded into ``visited`` so a staged copy is never read back from disk.
    The disk walk remains for the one reference class the staged closure
    cannot carry — a live (unstaged) include, in the root deck or nested
    inside a staged file — and for direct calls that pass no snapshots. It is
    bounded by the depth staging itself stages to
    (``DEFAULT_INCLUDE_DEPTH``): a chain shallow enough for staging to accept
    must not lint as missing. Disk resolution goes through deck staging's
    ``resolve_reference`` so the linter cannot read a different file than the
    one staging staged.
    """
    declared: set[str] = set()
    snapshot: dict[Path, list[SpiceCard]] = {}
    for path, text in context.includes:
        cards = lex(text).cards
        declared.update(_declared_models(cards))
        snapshot[path.resolve(strict=False)] = cards
    visited: set[Path] = set(snapshot)

    def walk(cards: list[SpiceCard], source: Path, depth: int) -> None:
        if depth >= DEFAULT_INCLUDE_DEPTH:
            return
        # ``depth`` here is the recursion budget; the scanner's parameter is
        # the library-context bit (0 = the deck itself, nonzero = a file
        # reached by following a reference). Keep the two separate.
        for reference in scan_include_references(cards, source, depth=min(depth, 1)):
            dependency = resolve_reference(source.parent, reference.raw_path)
            try:
                resolved = dependency.resolve(strict=False)
            except OSError:
                continue
            if resolved in visited:
                continue
            visited.add(resolved)
            if not resolved.is_file():
                continue
            try:
                nested = lex(read_spice_text(resolved)).cards
            except (OSError, UnicodeError, ValueError):
                continue
            declared.update(_declared_models(nested))
            walk(nested, resolved, depth + 1)

    walk(context.cards, context.path, 0)
    for staged_path, cards in snapshot.items():
        # Scanned from the snapshot, never from disk; the only thing this can
        # find that the snapshot itself does not carry is a live reference
        # nested inside a staged file.
        walk(cards, staged_path, 1)
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
                    "reason": (
                        "temperature is a simulator axis, not a parameter: a .param "
                        "named TEMP is never read as the simulation temperature, so "
                        "every point of a temperature sweep solves at the same "
                        "temperature and the rows come back identical. Set it with "
                        "'.temp <value ...>', '.step temp <list>', or "
                        "'.options temp=<value>' instead."
                    ),
                },
            )
        )
    return findings


_GENERATOR_RE = re.compile(r"(?i)^\*\s*generated by\s+(ltspice\b.*?)\s*$")


def deck_generator(text: str) -> str | None:
    """The writer a deck's ``* Generated by LTspice ...`` header names, if any.

    Only the first lines are read: the exporter writes the header at the top.
    """
    for line in text.splitlines()[:8]:
        match = _GENERATOR_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def value_suffix_evidence(
    site: ValueSuffixSite, *, generated_by: str | None = None
) -> dict[str, Any]:
    """What a non-ASCII suffix site means, shared by the linter and verify_circuit.

    ``reason`` explains; the other keys are facts: the token, the character's
    code point, and either the ASCII spelling of a micro sign or the number the
    simulator reads in place of a character that is no scale at all.
    """
    rest = site.token[len(site.number) + 1 :]
    evidence: dict[str, Any] = {
        "token": site.token,
        "suffix": f"U+{ord(site.suffix):04X}",
    }
    if site.micro:
        spelling = f"{site.number}u{rest}"
        evidence["ascii_spelling"] = spelling
        evidence["reason"] = (
            f"'{site.suffix}' is a micro suffix only to a reader that decodes this "
            "file in the encoding it was written in. In UTF-8, which LTspice 24 and "
            "later write, it is two bytes (µ is C2 B5); LTspice XVII decodes a deck "
            "as cp1252, reads them as two characters ('Âµ'), and drops the scale "
            f"without a diagnostic, so {site.token} runs as {site.number}. Write "
            f"{spelling}: 'u' is micro in every encoding and to every simulator."
        )
    else:
        evidence["reads_as"] = site.number
        if site.misdecoded_micro:
            intended = f"{site.number}u{rest[1:]}"
            evidence["likely_intended"] = intended
            evidence["reason"] = (
                f"'{site.token[len(site.number) : len(site.number) + 2]}' is a UTF-8 "
                "micro sign decoded as cp1252. Neither character is a scale "
                f"suffix, so the simulator reads {site.number}, a factor of 1e6 "
                f"from {intended}. Write {intended} if micro was meant."
            )
        else:
            evidence["reason"] = (
                f"'{site.suffix}' is not a scale suffix, so the simulator reads "
                f"{site.number} with no scale. If a scale was meant the value is "
                "wrong by that factor; if none was, drop the character. Scale "
                "suffixes are ASCII: f p n u m k meg g t."
            )
    if generated_by is not None:
        evidence["generated_by"] = generated_by
    return evidence


def _value_suffix_findings(
    context: _LintContext, rule: LintRule, *, micro: bool
) -> list[LintFinding]:
    return [
        _finding(
            context,
            rule,
            line=site.line,
            subject=site.token,
            file=path,
            evidence={
                **value_suffix_evidence(site, generated_by=generated_by),
                "directive": site.card.body,
            },
        )
        for path, generated_by, site in context.value_suffix_sites
        if site.micro is micro
    ]


def _value_suffix_micro_sign(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    return _value_suffix_findings(context, rule, micro=True)


def _value_suffix_nonascii(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    return _value_suffix_findings(context, rule, micro=False)


def _normalize_signal(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


RULES: tuple[LintRule, ...] = (
    LintRule("save-meas-coverage", "blocking", _save_meas_coverage),
    LintRule("meas-ngspice-batch", "blocking", _meas_ngspice_batch),
    LintRule("lib-section-ngspice", "blocking", _lib_section_ngspice),
    LintRule("model-missing", "blocking", _model_missing),
    LintRule("directive-arity", "blocking", _directive_arity),
    # A warning, not blocking: the run still answers a real question at the base
    # value, and the single-run path's hard refusal of the same deck is the
    # stricter reading of one behavior. What matters is that the caller learns
    # the sweep did not happen, since ngspice itself says nothing.
    LintRule("step-ngspice", "warning", _step_ngspice),
    LintRule("include-relative", "warning", _include_relative),
    LintRule("suffix-mega-milli", "warning", _suffix_mega_milli),
    # Blocking, not a warning: a .param TEMP does not set temperature, so the
    # deck simulates cleanly and returns one temperature's answers labelled as
    # several. That silent-wrong-answer class is what this linter exists to
    # stop, and a warning under the default lint mode does not stop it.
    LintRule("temp-as-param", "blocking", _temp_as_param),
    # A warning: a micro sign is micro to a reader that decodes the deck in
    # the encoding it was written in, and a staged deck spells it 'u' anyway.
    # The rule is for a deck that will be run somewhere else.
    LintRule("value-suffix-micro-sign", "warning", _value_suffix_micro_sign),
    # Blocking: any other non-ASCII character is never a scale suffix, so the
    # deck runs at the bare number. 'Âµ' — a UTF-8 micro sign decoded as
    # cp1252 — lands here and is a factor of 1e6 off.
    LintRule("value-suffix-nonascii", "blocking", _value_suffix_nonascii),
)

RULES_BY_ID: dict[str, LintRule] = {rule.rule_id: rule for rule in RULES}


def lint_deck(
    deck_text: str,
    path: Path,
    dialect: str | None,
    simulator: type | str | None,
    *,
    suppress: list[str] | set[str] | tuple[str, ...] = (),
    includes: Sequence[tuple[Path, str]] = (),
    ngbehavior: str | None = None,
) -> list[LintFinding]:
    """Run all unsuppressed rules and return fixable findings only.

    ``includes`` carries the staged include closure as (staged path, staged
    text) snapshots so rules resolve declarations through the snapshot
    instead of re-reading the deck's rewritten references from disk.
    """
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
        includes=tuple(includes),
        ngbehavior=ngbehavior,
    )
    findings: list[LintFinding] = []
    for rule in RULES:
        if rule.rule_id in suppressed:
            continue
        findings.extend(rule.check(context, rule))
    return findings
