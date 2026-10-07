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
from ltspice_mcp.lib.simulator import (
    SIMULATOR_DISPLAY,
    SIMULATORS,
    current_ngbehavior,
    simulator_family,
)
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_ops import MICRO_SIGN_READERS, ValueSuffixSite, value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import InstanceLine, MeasCard
from ltspice_mcp.lib.spice_validator import (
    ARITY_CHECKS,
    PROBE_REF_RE,
    drop_title_card,
    validate_netlist_arity,
)

Disposition = Literal["blocking", "warning", "observation"]
LintFinding = dict[str, Any]

linter_version = "5"

_SIGNAL_RE = PROBE_REF_RE
# A capital M straight after a number is milli unless the letters after it
# make it Meg or mil. Other letters do not: LTspice reads 1MHz as a millihertz.
_MILLI_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)M(?![Ee][Gg]|[Ii][Ll])\w*"
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
    # The deck's cards without its line-1 title. Both simulators skip line 1
    # of a netlist, so a title that starts with an element letter (``Diode
    # clamp test``) is prose, not a card, for every rule.
    cards: list[SpiceCard]
    # The simulator family the deck is linted for (see ``lint_deck``).
    family: str
    # The staged include closure, as (staged path, staged text) snapshots. On
    # a deck staged for a Windows simulator the rewritten references cannot be
    # re-read from the Linux side, so the snapshots are the authoritative
    # source for declarations the deck reaches through an include.
    includes: tuple[tuple[Path, str], ...] = ()
    ngbehavior: str | None = None

    @property
    def ngspice(self) -> bool:
        return self.family == "ngspice"

    @functools.cached_property
    def arity_issues(self) -> list[dict[str, object]]:
        """``validate_netlist_arity`` over the deck, run once for every arity rule.

        Its one simulator-specific check (a C=/L= primary value) is LTspice's, so
        only an LTspice deck is held to it.
        """
        return validate_netlist_arity(self.cards, simulator=SIMULATOR_DISPLAY[self.family])

    @functools.cached_property
    def include_cards(self) -> tuple[tuple[Path, str, list[SpiceCard]], ...]:
        """Each staged include snapshot with its cards, lexed once for every rule."""
        return tuple((path, text, lex(text).cards) for path, text in self.includes)


# The severity a finding reports for its rule's disposition.
DISPOSITION_SEVERITY: dict[Disposition, str] = {
    "blocking": "error",
    "warning": "warning",
    "observation": "observation",
}


def rule_severity(rule_id: str) -> str:
    """The severity a finding of ``rule_id`` reports."""
    return DISPOSITION_SEVERITY[RULES_BY_ID[rule_id].disposition]


def _finding(
    context: _LintContext,
    rule: LintRule,
    *,
    line: int,
    subject: str,
    evidence: Any,
    file: Path | None = None,
) -> LintFinding:
    return {
        "rule_id": rule.rule_id,
        "severity": DISPOSITION_SEVERITY[rule.disposition],
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
                "reason": (
                    "ngspice batch mode with a raw output skips top-level .meas: "
                    "the deck runs but this measurement is not evaluated, and "
                    "reading the run relays ngspice's notice of the skip. Measure "
                    "the waveform with analyze_results (signal_stats or value), or "
                    "inside a .control block."
                ),
            },
        )
        for card in context.cards
        if card.kind == "meas" and card.scope == ()
    ]


#: The functions whose angle LTspice's .meas evaluator reads or returns in the
#: unit its RadianMeasure setting names, which is degrees on the defaults of
#: LTspice 26 and XVII. A B source uses radians. The hyperbolic functions take
#: no angle and give the same value in either unit.
MEAS_ANGLE_FUNCTIONS = frozenset({"sin", "cos", "tan", "asin", "acos", "atan", "atan2"})

MEAS_ANGLE_REASON = (
    "On the default settings of LTspice 26 and XVII, sin, cos, tan, asin, acos, "
    "atan and atan2 take and give degrees inside a .meas and radians inside a "
    "B source: atan2(1,1) is 45 in a .meas and 0.785398 in a B source, and "
    "cos(pi) is 0.998497 against -1. The .meas unit is the per-user setting "
    "'Use radian measure in waveform expressions' (RadianMeasure), so the deck "
    "does not decide it. Compute the expression in a B source and measure its "
    "node (B1 x 0 V=V(out)*cos(2*pi*f*time), then .meas tran r INTEG V(x)), or "
    "combine the measured values after the run."
)


def meas_angle_functions(card: SpiceCard) -> list[str]:
    """The angle functions a ``.meas`` card calls, each once, in the order written."""
    try:
        meas = MeasCard.from_card(card)
    except SpiceLexError:
        return []
    names: list[str] = []
    for call in meas.function_calls:
        name = call.name.casefold()
        if name in MEAS_ANGLE_FUNCTIONS and name not in names:
            names.append(name)
    return names


def _meas_trig_degrees(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    if context.family != "ltspice":
        return []
    findings = []
    for card in context.cards:
        if card.kind != "meas" or card.scope != ():
            continue
        functions = meas_angle_functions(card)
        if not functions:
            continue
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=card.name or ".meas",
                evidence={
                    "functions": functions,
                    "directive": card.body,
                    "reason": MEAS_ANGLE_REASON,
                },
            )
        )
    return findings


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
                        "this compatibility mode treats a sectioned .lib as plain "
                        "includes and drops the section, so ngspice cannot find the "
                        'file. Set [simulator] ngbehavior = "hsa" in the server '
                        "config (or LTSPICE_MCP_NGBEHAVIOR=hsa) and restart the "
                        "server; ngspice then loads the section."
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
    for path, _text, cards in context.include_cards:
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


def _arity_check(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """The issues of the one ``validate_netlist_arity`` check named ``rule_id``."""
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
        for issue in context.arity_issues
        if issue["check"] == rule.rule_id
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
    The header ends in a full stop the build's own name does not have, so it
    is left off and the name reads as the log's banner does. LTspice XVII
    writes no such line.
    """
    for line in text.split("\n", 8)[:8]:
        match = _GENERATOR_RE.match(line.strip())
        if match:
            return match.group(1).rstrip(".")
    return None


def value_suffix_evidence(site: ValueSuffixSite, *, generated_by: str | None) -> dict[str, Any]:
    """What a non-ASCII suffix site means, shared by the linter and verify_circuit.

    ``reason`` explains; the other keys are facts: the token, the character's
    code point, and either the ASCII spelling of a micro sign or the number the
    simulator reads in place of a character that is no scale at all.
    """
    evidence: dict[str, Any] = {
        "token": site.token,
        "suffix": f"U+{ord(site.suffix):04X}",
    }
    if site.micro:
        spelling = f"{site.number}u{site.tail}"
        evidence["ascii_spelling"] = spelling
        evidence["reason"] = (
            f"'{site.suffix}' is a micro suffix only to a reader that decodes this "
            f"file in the encoding it was written in. {MICRO_SIGN_READERS} Misread, "
            f"{site.token} runs as {site.number}; write {spelling}."
        )
    else:
        evidence["reads_as"] = site.number
        if site.misdecoded_micro:
            intended = f"{site.number}u{site.tail[1:]}"
            evidence["likely_intended"] = intended
            evidence["reason"] = (
                f"'{site.suffix}{site.tail[:1]}' is a UTF-8 "
                "micro sign decoded as cp1252. Neither character is a scale "
                f"suffix, so the simulator reads {site.number}, a factor of 1e6 "
                f"from {intended}. Write {intended} if micro was meant."
            )
        elif site.mojibake:
            evidence["reason"] = (
                f"'{site.suffix}' is how cp1252 shows the first byte of a "
                "multi-byte UTF-8 character, so this text was decoded in an "
                "encoding it was not written in. It is not a scale suffix, so the "
                f"simulator reads {site.number}; if a micro sign was damaged this "
                "way, that is a factor of 1e6. Write the suffix in ASCII: "
                "f p n u m k meg g t."
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
    context: _LintContext, rule: LintRule, *, mojibake: bool
) -> list[LintFinding]:
    """Non-ASCII suffix sites in the deck and its staged includes.

    ``mojibake`` picks the sites whose suffix shows a mis-decoded file, or
    every other one. A micro sign itself is neither: staging has spelled it
    'u' by the time the deck is linted.
    """
    findings: list[LintFinding] = []
    files = [(context.path, context.text, context.cards), *context.include_cards]
    for path, text, cards in files:
        sites = [
            site
            for site in value_suffix_sites(cards)
            if not site.micro and site.mojibake == mojibake
        ]
        if not sites:
            continue
        generated_by = deck_generator(text)
        findings.extend(
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
            for site in sites
        )
    return findings


def _normalize_signal(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


_ARITY_DISPOSITION: dict[str, Disposition] = {
    severity: disposition for disposition, severity in DISPOSITION_SEVERITY.items()
}

RULES: tuple[LintRule, ...] = (
    LintRule("save-meas-coverage", "blocking", _save_meas_coverage),
    # A warning, not blocking: ngspice runs the deck and skips only the
    # top-level .meas, and reading the run relays ngspice's own notice of the
    # skip. Refusing the deck would cost the caller the rest of the run.
    LintRule("meas-ngspice-batch", "warning", _meas_ngspice_batch),
    # Blocking: on both builds' defaults a .meas that calls a trig function
    # runs cleanly and reads its angle in degrees, where the same expression
    # in a B source is in radians, and a user setting outside the deck can
    # change that unit. The number that comes back is wrong without any sign
    # in the log, which a warning under the default lint mode does not stop.
    LintRule("meas-trig-degrees", "blocking", _meas_trig_degrees),
    LintRule("lib-section-ngspice", "blocking", _lib_section_ngspice),
    LintRule("model-missing", "blocking", _model_missing),
    # One rule per validate_netlist_arity check, each at the disposition its
    # declared severity names, so suppressing one never silences another.
    *(
        LintRule(check, _ARITY_DISPOSITION[severity], _arity_check)
        for check, severity in ARITY_CHECKS.items()
    ),
    # A warning, not blocking: the run still answers a real question at the base
    # value. What matters is that the caller learns the sweep did not happen,
    # since ngspice itself says nothing.
    LintRule("step-ngspice", "warning", _step_ngspice),
    LintRule("include-relative", "warning", _include_relative),
    LintRule("suffix-mega-milli", "warning", _suffix_mega_milli),
    # Blocking, not a warning: a .param TEMP does not set temperature, so the
    # deck simulates cleanly and returns one temperature's answers labelled as
    # several. That silent-wrong-answer class is what this linter exists to
    # stop, and a warning under the default lint mode does not stop it.
    LintRule("temp-as-param", "blocking", _temp_as_param),
    # Blocking: a suffix that shows a file decoded in an encoding it was not
    # written in. 'Âµ' — a UTF-8 micro sign decoded as cp1252 — is never a
    # scale, so the deck runs at the bare number, a factor of 1e6 off. A micro
    # sign itself never reaches the linter: staging has spelled it 'u' by then,
    # and verify_circuit reports it for a deck that will run elsewhere, as a
    # warning only where a reader it knows of decodes the file otherwise.
    LintRule(
        "value-suffix-mojibake",
        "blocking",
        functools.partial(_value_suffix_findings, mojibake=True),
    ),
    # A warning: any other symbol after a number ('10Ω', '25°C') is read as the
    # bare number, which is usually what it means.
    LintRule(
        "value-suffix-nonascii",
        "warning",
        functools.partial(_value_suffix_findings, mojibake=False),
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
    includes: Sequence[tuple[Path, str]] = (),
    ngbehavior: str | None = None,
) -> list[LintFinding]:
    """Run all unsuppressed rules and return fixable findings only.

    ``includes`` carries the staged include closure as (staged path, staged
    text) snapshots so rules resolve declarations through the snapshot
    instead of re-reading the deck's rewritten references from disk.
    """
    suppressed = set(suppress)
    # A raw dialect names its family for every simulator but LTspice, so a
    # non-LTspice dialect decides; otherwise the simulator class (or recorded
    # class name) does, and a deck nothing names is linted as LTspice's.
    named_by_dialect = dialect if dialect in SIMULATORS and dialect != "ltspice" else None
    context = _LintContext(
        text=deck_text,
        path=path,
        cards=drop_title_card(lex(deck_text).cards),
        family=named_by_dialect or simulator_family(simulator) or "ltspice",
        includes=tuple(includes),
        ngbehavior=ngbehavior,
    )
    findings: list[LintFinding] = []
    for rule in RULES:
        if rule.rule_id in suppressed:
            continue
        findings.extend(rule.check(context, rule))
    return findings
