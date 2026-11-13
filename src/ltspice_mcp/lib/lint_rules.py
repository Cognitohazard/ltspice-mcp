"""Deterministic preflight lint registry for experiment decks."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from ltspice_mcp.lib.deck_staging import (
    DEFAULT_INCLUDE_DEPTH,
    is_absolute_reference,
    resolve_reference,
    scan_include_references,
    unreadable_references,
)
from ltspice_mcp.lib.encoding import latin1_reading, read_spice_text
from ltspice_mcp.lib.families import DISPLAY
from ltspice_mcp.lib.ngspice_compat import drops_lib_sections, reads_lib_as_includes
from ltspice_mcp.lib.simulator import DEFAULT_FAMILY, family_for, run_ngbehavior
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    directive_head,
    lex,
    lex_deck,
    unreadable_cards,
)
from ltspice_mcp.lib.spice_lex_ops import MICRO_SIGN_READERS, ValueSuffixSite, value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    MeasCard,
    ProbeRef,
    assigned_params,
    probe_is_pattern,
    probe_refs,
    read_instance,
    save_cards,
)
from ltspice_mcp.lib.spice_validator import (
    ARITY_CHECKS,
    EXCLUSIVE_ANALYSIS_KINDS,
    is_ground_node,
    may_connect_nodes,
    meas_functions_refused,
    meas_name_refused,
    meas_unclosed_paren_refused,
    step_refused,
    validate_netlist_arity,
    validate_netlist_duplicate_params,
)

Disposition = Literal["blocking", "warning", "observation"]
LintFinding = dict[str, Any]

linter_version = "14"

# A capital M straight after a number is milli unless the letters after it
# make it Meg or mil. Other letters do not: LTspice reads 1MHz as a millihertz.
_MILLI_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)M(?![Ee][Gg]|[Ii][Ll])\w*"
)
_MODEL_PREFIXES = frozenset({"D", "M", "Q", "J", "X", "S", "W"})

RuleCheck = Callable[["_LintContext", "LintRule"], list[LintFinding]]


@dataclass(frozen=True)
class LintRule:
    """Metadata and implementation for one deterministic lint rule.

    ``family`` is the simulator family whose reading of a deck the rule
    models, None for a rule that holds for every family. ``lint_deck`` runs a
    rule only for its family, and the capability table (``capabilities``)
    says a family has lint rules only when a rule declares it.
    """

    rule_id: str
    disposition: Disposition
    check: RuleCheck
    family: str | None = None


class _LintFile(NamedTuple):
    """One file a rule reads: the deck or a staged include, lexed once."""

    path: Path
    text: str
    cards: list[SpiceCard]
    # The codec it was decoded in (``StagedFile.codec``). Only an 8-bit file
    # (``cp1252``) can hold a byte UTF-8 would not.
    codec: str


@dataclass(frozen=True)
class _LintContext:
    text: str
    path: Path
    # The deck's cards the simulator reads (``spice_lex.lex_deck``): not its
    # line-1 title, which is prose whatever it says (``Diode clamp test``),
    # nor a card after ``.END``, so no rule refuses a card no run reads.
    cards: list[SpiceCard]
    # The simulator family the deck is linted for (see ``lint_deck``).
    family: str
    # The staged include closure, as (staged path, staged text, codec)
    # snapshots. On a deck staged for a Windows simulator the rewritten
    # references cannot be re-read from the Linux side, so the snapshots are
    # the authoritative source for declarations the deck reaches through an
    # include.
    includes: tuple[tuple[Path, str, str], ...] = ()
    ngbehavior: str | None = None
    # The codec the deck itself was decoded in.
    codec: str = "utf-8"
    # An LTspice the session knows reads decks as cp1252 (XVII or earlier),
    # with the evidence; None when nothing shows one.
    cp1252_reader: str | None = None

    @functools.cached_property
    def arity_issues(self) -> list[dict[str, object]]:
        """``validate_netlist_arity`` over the deck, run once for every arity rule.

        Its one simulator-specific check (a C=/L= primary value) is LTspice's, so
        only an LTspice deck is held to it.
        """
        return validate_netlist_arity(self.cards, simulator=DISPLAY[self.family])

    @functools.cached_property
    def reached_models(self) -> _ReachedModels:
        """The models the deck reaches through its includes, read once for
        model-missing and the unreadable-card rule."""
        return _models_from_staged_dependencies(self)

    @functools.cached_property
    def files(self) -> tuple[_LintFile, ...]:
        """The deck, then each staged include snapshot, lexed once for every rule."""
        return (
            _LintFile(self.path, self.text, self.cards, self.codec),
            *(
                _LintFile(path, text, lex(text).cards, codec)
                for path, text, codec in self.includes
            ),
        )


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


def _probe_signals(probe: ProbeRef) -> set[str]:
    """The signals a probe asks for in a ``.meas``, or saves in a ``.save``.

    A voltage between two nodes is its two nodes: both LTspice builds measure
    ``V(a,b)`` from a saved ``V(a)`` and ``V(b)``, and ``.save V(a,b)`` saves
    ``V(a)`` and ``V(b)`` (recorded deck/save_pair_both_nodes and
    deck/save_pair_as_pair). Any other probe stands as written, a voltage
    against ground among them: neither build measures ``V(a,0)`` with only
    ``V(a)`` saved (deck/save_pair_to_ground).
    """
    if (
        probe.kind == "V"
        and len(probe.names) == 2
        and not any(is_ground_node(name) for name in probe.names)
    ):
        return {_normalize_signal(f"V({name})") for name in probe.names}
    return {_normalize_signal(probe.text)}


class _Unsaved(NamedTuple):
    """What a ``.meas`` reads that no ``.save`` names: ``missing`` when the
    ``.save`` list names its signals one by one, ``unchecked`` when it also
    names some by a pattern (``V(*)``), since which signals a pattern saves,
    of either kind, is not recorded."""

    card: SpiceCard
    missing: list[str]
    unchecked: list[str]
    saved: list[str]
    patterns: list[str]


def _unsaved(context: _LintContext) -> list[_Unsaved]:
    """For each ``.meas`` of a deck with a ``.save`` list, what it reads that
    the list does not name: the one reading of a ``.save``'s coverage both
    rules below report."""
    saves = save_cards(context.cards)
    measurements = [card for card in context.cards if card.kind == "meas"]
    if not saves or not measurements:
        return []
    # A .save no rule could split into parts (``unreadable_cards``) may save
    # any signal, so none is reported missing on a guess about it.
    if any(save.saves_all for save in saves) or unreadable_cards(save.card for save in saves):
        return []
    probes = [probe for save in saves for probe in save.probes]
    patterns = [probe for probe in probes if probe_is_pattern(probe)]
    saved = {
        signal
        for probe in probes
        if not probe_is_pattern(probe)
        for signal in _probe_signals(probe)
    }
    found = []
    for card in measurements:
        required = {signal for probe in probe_refs(card) for signal in _probe_signals(probe)}
        unnamed = sorted(required - saved)
        missing, unchecked = ([], unnamed) if patterns else (unnamed, [])
        if unnamed:
            found.append(
                _Unsaved(card, missing, unchecked, sorted(saved), [p.text for p in patterns])
            )
    return found


def _save_meas_coverage(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    return [
        _finding(
            context,
            rule,
            line=unsaved.card.line_start,
            subject=unsaved.card.name or ".meas",
            evidence={
                "missing_signals": unsaved.missing,
                "saved_signals": unsaved.saved,
                "directive": unsaved.card.body,
            },
        )
        for unsaved in _unsaved(context)
        if unsaved.missing
    ]


def _save_pattern_unrecorded(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    return [
        _finding(
            context,
            rule,
            line=unsaved.card.line_start,
            subject=unsaved.card.name or ".meas",
            evidence={
                "unchecked_signals": unsaved.unchecked,
                "patterns": unsaved.patterns,
                "directive": unsaved.card.body,
                "reason": (
                    "A .save names signals by a pattern, and which signals a pattern "
                    "saves is not recorded, so these were not checked. Name them in "
                    "the .save to have them checked."
                ),
            },
        )
        for unsaved in _unsaved(context)
        if unsaved.unchecked
    ]


def _meas_ngspice_batch(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
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


def meas_angle_sites(cards: list[SpiceCard]) -> list[tuple[SpiceCard, list[str]]]:
    """Each top-level ``.meas`` card that calls an angle function, with the
    functions it calls, each once, in the order written.

    One scan for the linter's ``meas-trig-degrees`` rule and verify_circuit's
    syntax check, so both report the same cards.
    """
    sites = []
    for card in cards:
        if card.kind != "meas" or card.scope != ():
            continue
        try:
            calls = MeasCard.from_card(card).function_calls
        except SpiceLexError:
            continue
        names = (call.name.casefold() for call in calls)
        functions = list(dict.fromkeys(name for name in names if name in MEAS_ANGLE_FUNCTIONS))
        if functions:
            sites.append((card, functions))
    return sites


def _meas_trig_degrees(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    return [
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
        for card, functions in meas_angle_sites(context.cards)
    ]


def _step_ngspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    findings = []
    for card in context.cards:
        if card.kind != "directive" or directive_head(card) != ".step":
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


def _sectioned_libs(context: _LintContext) -> list[tuple[SpiceCard, str]]:
    """The deck's ``.lib file section`` cards and their sections, as deck
    staging reads them."""
    return [
        (reference.card, reference.section)
        for reference in scan_include_references(context.cards, context.path)
        if reference.section is not None
    ]


def _lib_section_ngspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    mode = run_ngbehavior(mode=context.ngbehavior)
    if not drops_lib_sections(mode):
        return []
    cause = (
        "treats a sectioned .lib as plain includes and drops the section, so ngspice "
        "cannot find the file"
        if reads_lib_as_includes(mode)
        else "(s3) does not read .lib at all, so the section's names are undefined"
    )
    findings = []
    for card, section in _sectioned_libs(context):
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=section,
                evidence={
                    "directive": card.body,
                    "ngbehavior": mode,
                    "reason": (
                        f"this compatibility mode {cause}. Set [simulator] "
                        'ngbehavior = "hsa" in the server config (or '
                        "LTSPICE_MCP_NGBEHAVIOR=hsa) and restart the server; ngspice "
                        "then loads the section."
                    ),
                },
            )
        )
    return findings


def _analysis_count_ltspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """LTspice runs one analysis a deck, with ``.op`` allowed beside it; two
    stop the run ("More than one analysis specified." on LTspice 26, a fatal
    error on XVII, recorded on both)."""
    analyses = [
        (file.path, card)
        for file in context.files
        for card in file.cards
        if card.kind == "directive" and directive_head(card)[1:] in EXCLUSIVE_ANALYSIS_KINDS
    ]
    if len(analyses) < 2:
        return []
    path, second = analyses[1]
    return [
        _finding(
            context,
            rule,
            line=second.line_start,
            file=path,
            subject=directive_head(second),
            evidence={
                "directives": [card.body for _, card in analyses],
                "reason": (
                    "LTspice runs one analysis per deck (.op may sit beside it) and "
                    "refuses a deck with more. Keep one, and run the others as "
                    "separate circuits or a variation."
                ),
            },
        )
    ]


def _meas_function_ltspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """A ``.meas`` calling a function only LTspice's waveform viewer has: LTspice
    26 stops at the directive and takes no measurement, XVII fails that one
    (recorded on both)."""
    findings = []
    for card in context.cards:
        refused = meas_functions_refused(card, "LTspice") if card.kind == "meas" else []
        if not refused:
            continue
        function, error = refused[0]
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=function,
                evidence={
                    "directive": card.body,
                    "reason": f"{error.message} {error.suggestion}",
                },
            )
        )
    return findings


def _meas_name_ltspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """A ``.meas`` named for a constant of the expression evaluator (``e``, ``k``,
    ``pi``, ``q``): LTspice 26 refuses the deck, XVII skips that measurement
    (recorded on both)."""
    return [
        _finding(
            context,
            rule,
            line=card.line_start,
            subject=card.name or ".meas",
            evidence={"directive": card.body, "reason": f"{error.message} {error.suggestion}"},
        )
        for card in context.cards
        if card.kind == "meas" and (error := meas_name_refused(card, "LTspice")) is not None
    ]


def _step_without_values_ltspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """A ``.step`` with nothing to step, which both builds refuse before the
    run (``spice_validator.step_refused``)."""
    return [
        _finding(
            context,
            rule,
            line=card.line_start,
            subject=".step",
            evidence={"directive": card.body, "reason": f"{error.message} {error.suggestion}"},
        )
        for card in context.cards
        if card.kind == "directive" and (error := step_refused(card.body, "LTspice")) is not None
    ]


def _lib_section_ltspice(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """``.lib file section`` selects a section in ngspice. LTspice has no sections:
    it looks for a file named ``file section`` and stops (recorded on both
    builds)."""
    findings = []
    for card, section in _sectioned_libs(context):
        findings.append(
            _finding(
                context,
                rule,
                line=card.line_start,
                subject=section,
                evidence={
                    "directive": card.body,
                    "reason": (
                        "LTspice has no library sections: it reads the rest of the "
                        "line as one file name and stops when no such file exists. "
                        "Include the file the section's models are in, or a copy "
                        "holding only that section."
                    ),
                },
            )
        )
    return findings


def _model_missing(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    reached = context.reached_models
    if not reached.complete:
        # Which models an include it cannot read declares is not known; the
        # unreadable-card rule reports that card and that this rule was quiet.
        return []
    declared = _declared_models(context.cards) | reached.declared
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


class _ReachedModels(NamedTuple):
    """The models a deck reaches through its include references; whether
    every include card on the way could be read, so that they are all of
    them; and those in a live file that could not."""

    declared: set[str]
    complete: bool
    unread: list[tuple[Path, SpiceCard, SpiceLexError]]


def _models_from_staged_dependencies(context: _LintContext) -> _ReachedModels:
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

    ``unread`` lists each include or library card in a live file that cannot
    be split into parts: which file it names is not known, so neither are the
    models it declares. Staging refuses such a card in every file it stages,
    and the unreadable-card rule reports one in the deck or a staged file.
    """
    declared: set[str] = set()
    complete = True
    unread: list[tuple[Path, SpiceCard, SpiceLexError]] = []
    snapshot: dict[Path, list[SpiceCard]] = {}
    for file in context.files[1:]:
        declared.update(_declared_models(file.cards))
        snapshot[file.path.resolve(strict=False)] = file.cards
    visited: set[Path] = set(snapshot)

    def walk(cards: list[SpiceCard], source: Path, depth: int, *, live: bool) -> None:
        nonlocal complete
        if depth >= DEFAULT_INCLUDE_DEPTH:
            return
        if blocked := unreadable_references(cards):
            complete = False
            if live:
                unread.extend((source, card, error) for card, error in blocked)
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
            walk(nested, resolved, depth + 1, live=True)

    walk(context.cards, context.path, 0, live=False)
    for staged_path, cards in snapshot.items():
        # Scanned from the snapshot, never from disk; the only thing this can
        # find that the snapshot itself does not carry is a live reference
        # nested inside a staged file.
        walk(cards, staged_path, 1, live=False)
    return _ReachedModels(declared, complete, unread)


def _arity_check(
    context: _LintContext,
    rule: LintRule,
) -> list[LintFinding]:
    """The issues of the one ``validate_netlist_arity`` check named ``rule_id``."""
    return _issue_findings(
        context, rule, (issue for issue in context.arity_issues if issue["check"] == rule.rule_id)
    )


def _issue_findings(
    context: _LintContext, rule: LintRule, issues: Iterable[Mapping[str, object]]
) -> list[LintFinding]:
    """A finding of ``rule`` for each validator issue: its line, the directive
    it is on, and its message and suggestion."""
    return [
        _finding(
            context,
            rule,
            line=raw_line if isinstance(raw_line := issue.get("line", 1), int) else 1,
            subject=str(issue.get("directive", "instance")),
            evidence={"message": issue.get("message"), "suggestion": issue.get("suggestion")},
        )
        for issue in issues
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
        try:
            names = [name for name, _token in assigned_params(card)]
        except SpiceLexError:
            # The unreadable-card rule reports it; the names it sets are not known.
            continue
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


class UnreadableSite(NamedTuple):
    """A card no rule could split into its parts, the lint rule that reports
    it, and what the finding says."""

    card: SpiceCard
    rule_id: str
    detail: str


def unreadable_card_sites(cards: Iterable[SpiceCard], family: str) -> list[UnreadableSite]:
    """Each card no rule could split into its parts (``spice_lex.unreadable_cards``),
    with the rule that reports it for ``family`` and what the finding says.

    For LTspice, a ``.meas`` that ``meas_unclosed_paren_refused`` refuses is
    ``meas-unclosed-paren-ltspice``; ``verify_circuit`` and ``edit_schematic``
    refuse it through the same validator. Any other is ``unreadable-card``:
    what a simulator does with it is not recorded. One answer for the linter
    and ``verify_circuit``, which reports the same cards.
    """
    simulator = DISPLAY.get(family, family)
    sites = []
    for card, error in unreadable_cards(cards):
        refused = meas_unclosed_paren_refused(card, simulator)
        if family == "ltspice" and refused is not None:
            sites.append(
                UnreadableSite(
                    card,
                    "meas-unclosed-paren-ltspice",
                    f"{refused.message} {refused.suggestion}",
                )
            )
            continue
        fix = f" {error.suggestion[:1].upper()}{error.suggestion[1:]}." if error.suggestion else ""
        # Only a card that may connect nodes has its words counted as
        # connected and defined; a .save or .param that cannot be read is not.
        counted = (
            ", and the nodes and names on it count as connected and defined"
            if may_connect_nodes(card)
            else ""
        )
        detail = (
            f"{error.message}: the checks could not split this card into its parts, "
            f"so no rule read them{counted}.{fix}"
        )
        sites.append(UnreadableSite(card, "unreadable-card", detail))
    return sites


def _unreadable_card(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    """A card no rule could read, in the deck or a staged include, under the
    rule ``unreadable_card_sites`` files it under: the other rules leave it
    out, so this says so. An include card in a live file that cannot be read
    is reported too, since model-missing says nothing for it."""
    located = [
        (file.path, site)
        for file in context.files
        for site in unreadable_card_sites(file.cards, context.family)
    ]
    if rule.rule_id == "unreadable-card":
        quiet = " Which file it names is not known, so model-missing reported nothing."
        for path, card, _error in context.reached_models.unread:
            (site,) = unreadable_card_sites([card], context.family)
            located.append((path, site._replace(detail=site.detail + quiet)))
    return [
        _finding(
            context,
            rule,
            line=site.card.line_start,
            file=path,
            subject=" ".join(site.card.body.split()),
            evidence={"detail": site.detail},
        )
        for path, site in located
        if site.rule_id == rule.rule_id
    ]


def _duplicate_param(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    """An instance, ``.model`` or ``.subckt`` card giving one parameter more
    than once: which copy the simulator reads is not known here, so a value a
    caller set on one copy may not be the one that runs."""
    return _issue_findings(context, rule, validate_netlist_duplicate_params(context.cards))


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


# An LTspice export opens with the path of the schematic it came from. LTspice
# 24 and later follow it with the ``* Generated by`` line; XVII writes the path
# alone, so an export that names its schematic and no generator is XVII's or
# an earlier build's.
_EXPORT_SOURCE_RE = re.compile(r"(?i)^\*\s*(?:[a-z]:|\\\\)[^\r\n]*\.asc$")
UNNAMED_EXPORT_WRITER = "LTspice XVII or earlier (an export that names no generator)"


def export_writer(text: str) -> str | None:
    """The LTspice that exported a deck, as far as the deck itself shows.

    The build its ``* Generated by`` line names; otherwise, for a deck that
    opens with the path of its schematic as an export does,
    :data:`UNNAMED_EXPORT_WRITER`; otherwise None.
    """
    named = deck_generator(text)
    if named is not None:
        return named
    first = text.lstrip("\ufeff").split("\n", 1)[0].strip()
    return UNNAMED_EXPORT_WRITER if _EXPORT_SOURCE_RE.match(first) else None


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
    for path, text, cards, _codec in context.files:
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


# Byte 0x85 as the server reads an 8-bit file: cp1252's ellipsis. LTspice
# reads it as Latin-1's next-line control.
_BYTE_85 = bytes([0x85]).decode("cp1252")


def _byte_85_ltspice(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    """A byte 0x85 in an 8-bit file, with a card after it on the same line.

    LTspice 24 and later end the line at that byte, so what follows it is read
    as a card of its own; XVII reads one line, as the server does. The byte is
    an ellipsis typed in a Western editor, and half of many double-byte
    characters. Silent when the session's LTspice is known to be XVII.
    """
    if context.cp1252_reader is not None:
        return []
    findings: list[LintFinding] = []
    for path, text, _cards, codec in context.files:
        if codec != "cp1252" or _BYTE_85 not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            after = [part.strip() for part in line.split(_BYTE_85)[1:]]
            cards = [part for part in after if part and not part.startswith("*")]
            if not cards:
                continue
            findings.append(
                _finding(
                    context,
                    rule,
                    line=number,
                    subject=cards[0],
                    file=path,
                    evidence={
                        "line": line.strip(),
                        "read_as_cards": cards,
                        "reason": (
                            "byte 0x85, shown here as an ellipsis, ends the line for "
                            "LTspice 24 and later, which read an 8-bit file as Latin-1, "
                            "where it is the next-line control. What follows it is read "
                            "as a card of its own; LTspice XVII reads one line. Remove "
                            "the byte, or save the file as UTF-8."
                        ),
                    },
                )
            )
    return findings


def _holds_c1_control(text: str) -> bool:
    """``text`` holds a control character from U+0080 to U+009F."""
    return any("\x80" <= char <= "\x9f" for char in text)


def _node_control_byte_ltspice(context: _LintContext, rule: LintRule) -> list[LintFinding]:
    """A node name in an 8-bit file holding a byte from 0x80 to 0x9F.

    LTspice reads each such byte as a control character, where the server
    shows cp1252's character for most of them. LTspice 26 refuses the card
    ("Expected node name here."); XVII runs it and saves the name with the
    control character, which is not the name the server reads.
    """
    findings: list[LintFinding] = []
    for path, text, cards, codec in context.files:
        # One scan of the file before any card is read: most 8-bit files hold
        # only characters from 0xA0 up (a degree sign, a copyright).
        if codec != "cp1252" or not _holds_c1_control(latin1_reading(text)):
            continue
        for card in cards:
            if card.kind != "instance" or not _holds_c1_control(latin1_reading(card.body)):
                continue
            line = read_instance(card)
            for node in line.nodes if line is not None else ():
                saved = latin1_reading(node)
                if not _holds_c1_control(saved):
                    continue
                findings.append(
                    _finding(
                        context,
                        rule,
                        line=card.line_start,
                        subject=node,
                        file=path,
                        evidence={
                            "card": card.body,
                            "saved_as": saved.encode("unicode_escape").decode("ascii"),
                            "reason": (
                                f"node {node!r} holds a byte from 0x80 to 0x9F, which "
                                "LTspice reads as a control character in an 8-bit "
                                "file. LTspice 26 refuses the card; XVII runs it and "
                                "saves the node under that control character, so ask "
                                "for its signal by the saved name. Rename the node "
                                "with ASCII to avoid both."
                            ),
                        },
                    )
                )
    return findings


def _normalize_signal(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


_ARITY_DISPOSITION: dict[str, Disposition] = {
    severity: disposition for disposition, severity in DISPOSITION_SEVERITY.items()
}

RULES: tuple[LintRule, ...] = (
    LintRule("save-meas-coverage", "blocking", _save_meas_coverage),
    LintRule("save-pattern-unrecorded", "warning", _save_pattern_unrecorded),
    # A warning, not blocking: ngspice runs the deck and skips only the
    # top-level .meas, and reading the run relays ngspice's own notice of the
    # skip. Refusing the deck would cost the caller the rest of the run.
    LintRule("meas-ngspice-batch", "warning", _meas_ngspice_batch, family="ngspice"),
    # Blocking: on both builds' defaults a .meas that calls a trig function
    # runs cleanly and reads its angle in degrees, where the same expression
    # in a B source is in radians, and a user setting outside the deck can
    # change that unit. The number that comes back is wrong without any sign
    # in the log, which a warning under the default lint mode does not stop.
    LintRule("meas-trig-degrees", "blocking", _meas_trig_degrees, family="ltspice"),
    LintRule("lib-section-ngspice", "blocking", _lib_section_ngspice, family="ngspice"),
    # Blocking: LTspice refuses each of these decks before it runs, or, on XVII
    # for a .meas function or name, fails the measurement the caller asked for.
    LintRule("analysis-count-ltspice", "blocking", _analysis_count_ltspice, family="ltspice"),
    LintRule("meas-function-ltspice", "blocking", _meas_function_ltspice, family="ltspice"),
    LintRule("meas-name-ltspice", "blocking", _meas_name_ltspice, family="ltspice"),
    LintRule("meas-unclosed-paren-ltspice", "blocking", _unreadable_card, family="ltspice"),
    LintRule("lib-section-ltspice", "blocking", _lib_section_ltspice, family="ltspice"),
    LintRule(
        "step-without-values-ltspice", "blocking", _step_without_values_ltspice, family="ltspice"
    ),
    # A warning, not blocking: LTspice 24 and later run another circuit than
    # the one the server reads, XVII the same one, and which build will run
    # the deck is not always known here.
    LintRule("byte-85-ltspice", "warning", _byte_85_ltspice, family="ltspice"),
    # A warning: LTspice 26 refuses the card and says so in its log, while
    # XVII runs it under a name the server does not spell the same way.
    LintRule("node-control-byte-ltspice", "warning", _node_control_byte_ltspice, family="ltspice"),
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
    LintRule("step-ngspice", "warning", _step_ngspice, family="ngspice"),
    LintRule("include-relative", "warning", _include_relative),
    LintRule("suffix-mega-milli", "warning", _suffix_mega_milli),
    # Blocking, not a warning: a .param TEMP does not set temperature, so the
    # deck simulates cleanly and returns one temperature's answers labelled as
    # several. That silent-wrong-answer class is what this linter exists to
    # stop, and a warning under the default lint mode does not stop it.
    LintRule("temp-as-param", "blocking", _temp_as_param),
    # A warning: the deck runs, but a parameter given twice is read one way or
    # the other, and which is not recorded here.
    LintRule("duplicate-param", "warning", _duplicate_param),
    # A warning: a card no rule could split into its parts is checked by
    # none, which a caller must know. What a simulator does with it is not
    # recorded, except for the .meas above, so the deck runs.
    LintRule("unreadable-card", "warning", _unreadable_card),
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
    includes: Sequence[tuple[Path, str, str]] = (),
    ngbehavior: str | None = None,
    codec: str = "utf-8",
    cp1252_reader: str | None = None,
) -> list[LintFinding]:
    """Run all unsuppressed rules and return fixable findings only.

    ``includes`` carries the staged include closure as (staged path, staged
    text, codec) snapshots so rules resolve declarations through the snapshot
    instead of re-reading the deck's rewritten references from disk. ``codec``
    is the one the deck was decoded in, as each snapshot's is
    (``StagedFile.codec``); ``cp1252_reader`` is the evidence that the
    session's LTspice reads decks as cp1252 (``services.cp1252_ltspice``).
    """
    suppressed = set(suppress)
    context = _LintContext(
        text=deck_text,
        path=path,
        cards=lex_deck(deck_text).cards,
        # A deck nothing names is linted as the default family's.
        family=family_for(dialect, simulator) or DEFAULT_FAMILY,
        includes=tuple(includes),
        ngbehavior=ngbehavior,
        codec=codec,
        cp1252_reader=cp1252_reader,
    )
    findings: list[LintFinding] = []
    for rule in RULES:
        if rule.rule_id in suppressed:
            continue
        if rule.family is not None and rule.family != context.family:
            continue
        findings.extend(rule.check(context, rule))
    return findings
