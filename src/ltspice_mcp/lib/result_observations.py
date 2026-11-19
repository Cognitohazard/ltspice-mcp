"""Result observation surfacing: report the facts, do not rate the result.

The consumer of a simulation result here is an LLM agent with its own physics
knowledge: it already knows 1e30 V is a floating node. So this layer does NOT
render a trust verdict (a ``confidence: degraded`` label would inject our
subjective judgment in place of the model's better-informed one, and a wrong
label is a false accusation the model may parrot or learn to ignore). Instead it
*surfaces facts* the model would otherwise miss or have to dig for, and lets the
model judge.

This module holds the project's rules for reporting a result: report the
facts, do not rate them. The rules:
- **Severity is relayed, never invented.** Relay observations carry the
  simulator's own classification (it called it an error). Value observations
  carry none — just the fact and its evidence; the magnitude speaks for itself.
- **Surfacing dissolves the false-positive problem.** A surfaced fact is never a
  false accusation — at worst it is one redundant true fact the model dismisses
  cheaply. So the salience thresholds below decide only "is this worth lifting
  into view", not "is this wrong"; being slightly off is nearly free.
- **No ``ok`` signal.** Absence of observations means "nothing tripped a check",
  never "verified correct". Coverage gaps (a scan we skipped) are themselves
  surfaced as observations so an empty-looking result can't masquerade as a
  clean bill of health.

The output is ``list[Observation]`` (empty when nothing was surfaced). It rides
on the run summary as ``summary["observations"]`` and is the curated "look here"
layer above the always-available raw ``warnings``/``errors`` lists.

Two surfacing channels, deliberately kept distinct (do not merge them):

- ``observations`` answer *is the underlying data/solve trustworthy?* — facts
  about the result and the simulator's own behavior (a relayed log error, a
  non-finite or rail-pinned value, a coverage gap). Structured, code-tagged,
  governed by the rules in this module.
- ``warnings`` (on the analysis tools — ``signal_stats``, ``edge_metrics``,
  ``bode_metrics``, ...) answer *did this measurement have to make assumptions?*
  — per-request caveats about the measurement just performed (a clamped window,
  an ambiguous edge, a Hann-window approximation), free-text and actionable.

These answer different questions and call for different consumer responses, so a
single value can legitimately appear in different channels across tools. A
*run-level solve failure* (singular matrix, non-convergence) is a data/solve
fact: it goes to ``observations`` where a tool has that channel, and otherwise
to that tool's ``warnings`` list (the channel it already surfaces). Folding the
two channels into one would conflate "your data is garbage" with "I approximated
your window" and force bespoke free-text guidance into a code taxonomy. The
metric tools simply lack an ``observations`` channel today; add one there (and
move their solve-failure relay onto it) only when a second data-fact needs a
home — not via a breaking ``warnings``→``observations`` migration.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.deck_staging import (
    DEFAULT_INCLUDE_DEPTH,
    resolve_existing,
    resolve_reference,
    scan_include_references,
)
from ltspice_mcp.lib.device_names import DevicePath, path_text, spelled_name, split_path, trace_key
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_conditions import said_lines
from ltspice_mcp.lib.log_types import (
    MEASUREMENTS_UNREAD,
    given_no_value,
    measurement_named,
    run_measurement_facts,
)
from ltspice_mcp.lib.ngspice_driver import interp_option, read_option_cards
from ltspice_mcp.lib.spice_lex import (
    EditCards,
    SpiceLexError,
    directive_head,
    extract_meas_name,
    lex_deck,
    lex_for_edit,
)
from ltspice_mcp.lib.spice_lex_views import directive_args, read_instance
from ltspice_mcp.lib.spice_validator import leading_numeric_tokens
from ltspice_mcp.lib.trace_names import is_node_voltage

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_raw import DecodedRaw
    from ltspice_mcp.lib.device_names import Naming
    from ltspice_mcp.lib.lint_rules import SaveList
    from ltspice_mcp.lib.netlist_graph import FlatNetlist
    from ltspice_mcp.lib.result_kinds import Analysis

# Salience threshold for lifting a node value into view. The gmin/floating-node
# artifact lands at ~1e30, so 1e8 has enormous headroom; a legitimate HV design
# may also clear it, but surfacing the value is a fact the model reconciles
# against intent, not an accusation — so a loose threshold is fine by design.
_EXTREME_VALUE_SALIENCE = 1e8

# Self-relative blow-up signature, for divergences that stay under the absolute
# floor (e.g. a switching node ringing to ~1e7 V on a 12 V rail). A divergence
# reaches a large magnitude that dwarfs the trace's own typical operating level,
# whatever fraction of the samples it occupies — so we compare the peak to the
# trace's median. Two gates keep it precise: the peak must clear an absolute
# floor (so a legitimate small-signal swing — a 0→12 V step has peak ≈ 12 — is
# never even considered), AND the ratio is large enough that only a node both
# reaching ≥10 kV and standing ~1e5× its own median trips it. The ratio is set
# below the cited buck's actual peak/median (~9e5 rail-dominant, ~1.8e6 idle-
# dominant) with margin, so detection does not hinge on which regime the median
# lands in. Rule: a relative signature, surfaced as a fact, not a verdict.
_EXTREME_VALUE_FLOOR = 1e4  # below this a spike is never "diverged", whatever the ratio
_EXTREME_VALUE_RATIO = 1e5  # peak ≥ this many × the trace's own median ⇒ blow-up signature

# Source-relative signature: a node voltage that dwarfs every independent
# voltage source in the deck (supply rails included, since those are V-cards
# too). Catches moderate-magnitude divergences the absolute/self-relative
# gates can't see — e.g. an undamped LC growing to ~1 kV from a 0.1 V drive
# stays far below the 1e4 absolute floor but is 10^4× the largest source.
# 20× keeps ordinary step-up quiet (switch-node ringing and boost ratios sit
# under ~10×) while still catching a moderate runaway measured against the
# supply rail — an 850 V runaway on a 12 V-railed deck is only ~71×, invisible
# at a 100× cutoff. Resonant/transformer step-up past 20× gets a true fact
# surfaced, which the model dismisses cheaply in context (rule above).
_SOURCE_RELATIVE_RATIO = 20.0  # peak |V| ≥ this many × the largest V-source amplitude

# Transient-function keywords a V-card may carry; used by the best-effort
# source-amplitude parser below.
_SOURCE_FUNCTION_KEYWORDS = frozenset({"sin", "sine", "pulse", "exp", "pwl", "sffm", "am"})


#: A compressed transient whose widest gap between stored points spans more
#: than this share of its time is coarser than twenty evenly spaced points.
_COARSE_GAP_SHARE = 1 / 20


class Observation(TypedDict, total=False):
    """One surfaced fact about a result.

    ``kind`` groups by where the fact came from (and how authoritative it is):
    ``relay`` (the simulator said so), ``reconciliation`` (you asked X, got Y),
    ``value`` (a number in the data), ``coverage`` (a check we did NOT run).
    ``severity`` is present ONLY for ``relay`` items — inherited from the
    simulator, never invented here.
    """

    code: str
    kind: str
    detail: str
    severity: str
    evidence: dict[str, Any]


#: A count on a ``.four`` card. LTspice's form is ``.four <freq> [Nharmonics]
#: [Nperiods] <trace> ...``, so up to two integers (Nperiods may be -1, the
#: whole run) sit between the frequency and the traces; a trace is never a
#: bare integer.
_FOUR_COUNT_RE = re.compile(r"[+-]?\d+")


def parse_requested_outputs(netlist_text: str) -> dict[str, list[str]]:
    """Extract requested ``.meas``/``.four`` names from a netlist deck.

    Returns ``{"meas": [name, ...], "four": [signal, ...]}`` preserving the
    original-case tokens. The deck is read as the simulator reads it
    (``spice_lex.lex_deck``): a card continued on ``+`` lines is one card, a
    comment is no part of it, and neither the title nor a card after ``.END``
    is read. A ``.meas`` name is the lexer's (``SpiceCard.name``), so
    reconciliation matches how produced names are derived elsewhere.
    Best-effort — reconciliation is advisory.
    """
    meas: list[str] = []
    four: list[str] = []
    for card in lex_deck(netlist_text).cards:
        if card.kind == "meas":
            if card.name:
                meas.append(card.name)
            continue
        if card.kind != "directive" or directive_head(card) not in (".four", ".fourier"):
            continue
        try:
            args = directive_args(card)
        except SpiceLexError:
            continue  # best effort: a card no rule reads names no output
        if args:
            rest = args[1:]
            counts = 0
            while counts < min(2, len(rest)) and _FOUR_COUNT_RE.fullmatch(rest[counts]):
                counts += 1
            four.extend(rest[counts:])
    return {"meas": meas, "four": four}


def _v_card_amplitude(spec: list[str]) -> float | None:
    """Peak |V| a voltage-source card can drive, from its value tokens.

    ``spec`` is the token list after ``Vname n+ n-`` with parentheses/commas
    already stripped. Returns the parsed peak drive level (0.0 for a card that
    demonstrably drives nothing — an AC-only source or a 0 V sense source), or
    ``None`` when the card could NOT be bounded (behavioral value, unresolved
    ``{param}``, PWL ``file=`` form, a function spec cut short by a
    non-numeric argument). The caller must treat None as poisoning the deck's
    ``max()`` — the dropped card might have been the largest source.
    """
    fn: str | None = None
    fn_args: list[float] = []
    spec_cut_short = False
    dc_val: float | None = None
    saw_ac = False
    i = 0
    while i < len(spec):
        tok = spec[i]
        if "=" in tok:  # Rser=1, Cpar=…, value={…} — not a drive level
            i += 1
            continue
        low = tok.lower()
        if low in _SOURCE_FUNCTION_KEYWORDS:
            fn = low
            # The function's argument run: consecutive numeric tokens after the
            # keyword. A non-numeric token ({param}, REPEAT, file=) ends it —
            # and any non-key=value remainder means the spec was cut short, so
            # the card can't be bounded.
            rest = spec[i + 1 :]
            for arg in rest:
                try:
                    fn_args.append(parse_spice_value(arg))
                except ValueError:
                    break
            spec_cut_short = any("=" not in t for t in rest[len(fn_args) :])
            break
        if low == "dc":
            i += 1
            continue  # the DC value itself is picked up as the next numeric
        if low == "ac":
            # Skip the small-signal spec (mag + optional phase). An AC-only
            # source drives nothing in .tran/.op, so it contributes no
            # amplitude of its own.
            saw_ac = True
            i += 1
            while i < len(spec):
                try:
                    parse_spice_value(spec[i])
                except ValueError:
                    break
                i += 1
            continue
        try:
            v = parse_spice_value(tok)
        except ValueError:
            return None  # a token we can't read might BE the drive level
        if dc_val is None:
            dc_val = v
        i += 1

    if fn is not None:
        if spec_cut_short:
            return None  # the missing argument may be the drive level
        if fn in ("sin", "sine", "sffm", "am"):
            # (offset, amplitude, …) — peak drive is |offset| + |amplitude|.
            return abs(fn_args[0]) + abs(fn_args[1]) if len(fn_args) >= 2 else None
        if fn in ("pulse", "exp"):
            # (V1, V2, …) — the drive swings between the two levels.
            return max(abs(fn_args[0]), abs(fn_args[1])) if len(fn_args) >= 2 else None
        # pwl: (t1, v1, t2, v2, …) — values sit at the odd positions.
        values = fn_args[1::2]
        return max(abs(v) for v in values) if values else None
    if dc_val is not None:
        return abs(dc_val)
    return 0.0 if saw_ac else None


_PARAM_ASSIGN_RE = re.compile(r"(\w+)\s*=\s*([^\s{}(),]+)")
_PARAM_REF_RE = re.compile(r"\{(\w+)\}\Z")


def _resolve_param_ref(tok: str, params: dict[str, float]) -> str:
    """A ``{name}`` reference to a literal ``.param`` constant becomes its
    value; any other token passes through unchanged."""
    m = _PARAM_REF_RE.match(tok)
    if m and m.group(1).lower() in params:
        return str(params[m.group(1).lower()])
    return tok


def parse_source_amplitudes(netlist_text: str) -> dict[str, float]:
    """Peak drive |V| of each independent voltage source in a deck, or ``{}``.

    Supply rails are V-cards too, so ``max()`` over the result is the deck's
    excitation scale — the reference the source-relative ``extreme_value``
    trigger compares node voltages against. Handles DC, SIN/SINE, PULSE, EXP,
    PWL, SFFM and AM specs, bare ``5V`` unit tails, and simple ``{name}``
    references to literal ``.param`` constants. The deck is read as the
    simulator reads it (``spice_lex.lex_deck``): a card continued on ``+``
    lines is one card, a comment is no part of it, and neither the title nor
    a card after ``.END`` is read.

    All-or-nothing: if ANY V-card cannot be bounded (behavioral, unresolvable
    ``{expr}``, a spec cut short, a card that cannot be read), the whole
    result is ``{}`` and the trigger stays disarmed — the dropped card might
    have been the deck's largest source, and a max() over the remainder would
    state a false "largest independent voltage source" fact. Quieter, never
    wrong.
    """
    cards = lex_deck(netlist_text).cards

    # Literal .param constants, for resolving {name} source values — the
    # common parameterized-rail idiom (.param vdd=12 / V1 vdd 0 {vdd}).
    # Non-numeric params (expressions) are simply absent from the map.
    params: dict[str, float] = {}
    for card in cards:
        if card.kind == "instance" or not card.body.lstrip().lower().startswith(".param"):
            continue
        for name, value in _PARAM_ASSIGN_RE.findall(card.body):
            try:
                params[name.lower()] = parse_spice_value(value)
            except ValueError:
                continue

    out: dict[str, float] = {}
    for card in cards:
        if card.kind != "instance" or not card.name or card.name[0] not in "Vv":
            continue
        line = read_instance(card)
        if line is None or len(line.nodes) < 2 or (line.value is None and line.source is None):
            # Unreadable, or a source that plays a .wav file, whose second word
            # is a node: it might be the largest source.
            return {}
        # The value is the source spec (``DC 5``, ``SIN(0 1 1k)``); its own
        # parameters after it (``Rser=1``, ``AC 2``) are never a drive level.
        # A source with no value drives nothing, an ``AC 1`` read as such.
        spec = line.value if line.value is not None else " ".join(line.own_words)
        tokens = spec.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if not tokens:
            continue
        amp = _v_card_amplitude([_resolve_param_ref(tok, params) for tok in tokens])
        if amp is None or not math.isfinite(amp):
            return {}  # an unbounded V-card poisons the max — disarm entirely
        if amp > 0:
            out[line.ref] = amp
    return out


def _first_directive(netlist_text: str, head: str) -> list[str] | None:
    """The words of a deck's first ``head`` directive (``.tran``, ``.ac``)
    that the simulator reads, its name first (``directive_args``); None when
    it has none, or when that card cannot be read."""
    for card in lex_deck(netlist_text).cards:
        if card.kind == "directive" and directive_head(card) == head:
            try:
                return [head, *directive_args(card)]
            except SpiceLexError:
                return None
    return None


def _read_deck(netlist: Path, parse: Callable[[str], float | None]) -> float | None:
    """``parse`` of the deck at ``netlist``; None when it cannot be read."""
    try:
        return parse(read_spice_text(netlist))
    except OSError:
        return None


def tran_max_step(netlist_text: str) -> float | None:
    """The maximum time step a deck's first ``.tran`` asks for, its fourth
    number (``.tran <Tstep> <Tstop> <Tstart> <Tmax>``). None when the card
    gives none, or a value that is not a number, or the deck has no ``.tran``."""
    tokens = _first_directive(netlist_text, ".tran")
    numbers = leading_numeric_tokens(tokens[1:5]) if tokens else []
    return numbers[3] if len(numbers) == 4 and numbers[3] > 0 else None


def deck_ngspice_interp(netlist: Path) -> tuple[bool | None, str | None]:
    """Whether the deck at ``netlist`` sets ngspice's ``interp`` option
    (``ngspice_driver.interp_option``) on a card ngspice reads
    (``ngspice_driver.read_option_cards``), in the deck or a file it
    ``.include``s.

    ``(True, None)`` when one does; ``(False, None)`` when every file the deck
    reads was read and no card it reads does; ``(None, why)`` otherwise: an
    option card setting it in a file read through ``.lib``, which part of such
    a file ngspice reads turning on its mode (a section, or the whole file);
    a ``.control`` block, which can set it as a command; a file that could not
    be read or lies deeper than ``deck_staging.DEFAULT_INCLUDE_DEPTH``."""
    files: list[EditCards] = []
    through_lib: list[bool] = []
    unread: str | None = None
    pending: list[tuple[Path, int, bool]] = [(netlist, 0, False)]
    seen: set[Path] = set()
    while pending:
        path, depth, lib = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            lexed = lex_for_edit(read_spice_text(path), deck=depth == 0)
            references = scan_include_references(lexed.read, path, family="ngspice", depth=depth)
        except (OSError, SpiceLexError):
            unread = unread or f"the file {path.name} could not be read"
            continue
        files.append(lexed)
        through_lib.append(lib)
        if any(card.kind == "control" for card in lexed.read):
            unread = unread or "a .control block can set it as a command, and is not read here"
        for reference in references:
            target = resolve_existing(resolve_reference(path.parent, reference.raw_path))
            if target is None:
                unread = unread or f"the file {reference.raw_path} the deck reads was not found"
            elif depth + 1 > DEFAULT_INCLUDE_DEPTH:
                unread = unread or (
                    f"the file {reference.raw_path} is more than {DEFAULT_INCLUDE_DEPTH} "
                    "includes deep, and is not read here"
                )
            else:
                pending.append((target, depth + 1, lib or reference.head == ".lib"))
    for index, card in read_option_cards(files):
        sets = interp_option(card)
        if sets is None:
            unread = unread or "an option card could not be read"
        elif sets and through_lib[index]:
            unread = unread or (
                "an option card in a file the deck reads through .lib sets it, and "
                "whether ngspice reads that card turns on the section and the mode"
            )
        elif sets:
            return True, None
    return (None, unread) if unread else (False, None)


def one_ac_frequency(netlist_text: str) -> float | None:
    """The one frequency a deck's first ``.ac`` names as ``.ac list <f>``: the
    frequency of a stepped ``.ac`` at one frequency, which its raw does not
    store (``result_kinds.STEPPED_AC_ONE_FREQUENCY``, recorded as
    ``raw/step_ac_one_freq``). None for any other form, or a value that is not
    a number."""
    tokens = _first_directive(netlist_text, ".ac")
    if tokens is None or len(tokens) != 3 or tokens[1].casefold() != "list":
        return None
    numbers = leading_numeric_tokens(tokens[2:])
    return numbers[0] if numbers and numbers[0] > 0 else None


#: Each deck's ``one_ac_frequency``, kept while the deck is unchanged: a
#: stepped run asks it once for every step it reads.
_ONE_AC_FREQUENCIES: FileCache[float | None] = FileCache(maxsize=64)


def deck_one_ac_frequency(netlist: Path) -> float | None:
    """``one_ac_frequency`` of the deck at ``netlist``; None when it cannot be read."""
    return _ONE_AC_FREQUENCIES.get(netlist, lambda path: _read_deck(path, one_ac_frequency))


#: Each deck's ``tran_max_step``, kept while the deck is unchanged.
_TRAN_MAX_STEPS: FileCache[float | None] = FileCache(maxsize=64)


def deck_tran_max_step(netlist: Path) -> float | None:
    """``tran_max_step`` of the deck at ``netlist``; None when it cannot be read."""
    return _TRAN_MAX_STEPS.get(netlist, lambda path: _read_deck(path, tran_max_step))


def compressed_transient(raw: DecodedRaw) -> bool:
    """Whether ``raw`` is an LTspice transient saved with waveform compression
    on, the plot ``coarse_compressed_waveform`` judges. LTspice adds
    ``nocompression`` to a transient's flags when compression is off
    (``raw/compress_*``). The flag shows the setting, not that points were
    dropped: LTspice 26 stores every point of ``raw/compress_1mv`` with it on.

    It reads the decoded plot, its dialect established, so it answers for the
    plot a read holds. ``raw_parser.may_hold_compressed_transient`` asks the
    same of the header alone, before anything is decoded, and answers True
    where it cannot tell, so a recipe that reads only the log decodes the raw
    only when it may need to. The two differ on purpose: one is a cheap,
    conservative pre-check, this is the answer."""
    descriptor = raw.descriptor
    return (
        descriptor.dialect == "ltspice"
        and descriptor.analysis == "transient"
        and not any(flag.lower() == "nocompression" for flag in raw.header.flags)
    )


def coarse_compressed_waveform(
    raw: DecodedRaw, *, max_step: float | None = None
) -> dict[str, float] | None:
    """The first step of a compressed LTspice transient whose widest gap
    between stored points spans more than a twentieth of its time, and more
    than ``max_step``, the step its deck asked the solver not to exceed: the
    step, its points and span, and where the gap lies. None for any other
    plot. A gap no wider than the deck's own maximum step is what it asked for.

    """
    if not compressed_transient(raw):
        return None
    try:
        for step in raw.get_steps():
            times = np.asarray(raw.get_axis(step))
            if times.size < 2:
                continue
            span = float(times[-1] - times[0])
            gaps = np.diff(times)
            widest = int(np.argmax(gaps))
            # A stored time can round past the step by a few parts in a billion.
            limit = max(span * _COARSE_GAP_SHARE, (max_step or 0.0) * (1 + 1e-6))
            if span > 0 and gaps[widest] > limit:
                return {
                    "step": step,
                    "points": times.size,
                    "span_s": span,
                    "gap_from_s": float(times[widest]),
                    "gap_to_s": float(times[widest + 1]),
                }
    except ResultError:
        return None  # step boundaries unresolved: no step to read
    return None


#: What the recordings show of LTspice's own ``.meas`` on a compressed
#: transient, by build (``raw/compress_1mv`` and ``raw/compress_1mv_off``).
_XVII_MEAS = (
    "LTspice XVII's own .meas results read points its raw does not hold: in a recorded "
    "RC charge they came within 0.25% of the uncompressed run's, while a trapezoid "
    "average over the 13 stored points was 0.41%, and interpolation at 1 ms 1.15%, "
    "below them. LTspice 26 stored every point of the same run."
)


def compressed_xvii_observation(labels: Sequence[str], *, run_cap: int) -> Observation:
    """The observation for LTspice XVII transient runs, each by its label,
    whose samples a read took from a raw saved with waveform compression on
    (``services.compression_drops_points``) and not already said to be coarse
    (``coarse_waveform_observation``)."""
    return Observation(
        code="compressed_transient",
        kind="value",
        detail=(
            f"{len(labels)} LTspice XVII transient run(s) read here were saved with "
            "waveform compression on, which in every recorded XVII run stored fewer "
            "points than it computed, so a value read from the samples is interpolated "
            f"across points the raw does not hold. {_XVII_MEAS} '.options plotwinsize=0' stores every "
            "point."
        ),
        evidence={"runs": list(labels[:run_cap]), "run_count": len(labels)},
    )


def coarse_waveform_observation(
    coarse: Sequence[tuple[str, Mapping[str, float]]], *, run_cap: int
) -> Observation:
    """The observation for the transient runs ``coarse_compressed_waveform``
    found coarse, each with its label: the first one's evidence, and up to
    ``run_cap`` labels."""
    label, first = coarse[0]
    span = first["span_s"]
    gap = first["gap_to_s"] - first["gap_from_s"]
    labels = [label for label, _ in coarse]
    return Observation(
        code="coarse_compressed_waveform",
        kind="value",
        detail=(
            f"{len(labels)} LTspice transient run(s) read here were saved with "
            f"waveform compression on and store few points. In {label}, "
            f"{first['points']:.0f} points span {span:.3g} s, and the widest gap "
            f"between two, from {first['gap_from_s']:.3g} s to "
            f"{first['gap_to_s']:.3g} s, is {gap / span:.0%} of that time. A value "
            f"read between stored points is interpolated across the gap. {_XVII_MEAS} "
            "'.options plotwinsize=0' stores every point."
        ),
        evidence={
            "run": label,
            **first,
            "runs": labels[:run_cap],
            "run_count": len(labels),
        },
    )


def deck_observation_inputs(
    netlist: Path,
) -> tuple[dict[str, list[str]] | None, dict[str, float] | None]:
    """Read a deck once and parse both observation inputs derived from it:
    the requested ``.meas``/``.four`` names (reconciliation) and the
    independent voltage-source amplitudes (source-relative trigger).

    The one call every summary path shares, so run-completion and later
    re-inspection feed the surfacer identically. ``(None, None)`` when the
    deck can't be read. Kept while the deck is unchanged, since a summary of
    each run, and of each step, asks again; each caller gets its own copy.
    """
    requested, amplitudes = _OBSERVATION_INPUTS.get(netlist, _observation_inputs)
    return (
        None if requested is None else {kind: list(names) for kind, names in requested.items()},
        None if amplitudes is None else dict(amplitudes),
    )


def _observation_inputs(
    netlist: Path,
) -> tuple[dict[str, list[str]] | None, dict[str, float] | None]:
    try:
        text = read_spice_text(netlist)
    except OSError:
        return None, None
    return parse_requested_outputs(text), parse_source_amplitudes(text)


#: Each deck's ``deck_observation_inputs``, kept while the deck is unchanged.
_OBSERVATION_INPUTS: FileCache[tuple[dict[str, list[str]] | None, dict[str, float] | None]] = (
    FileCache(maxsize=64)
)


def relay_observations(summary: dict, said: Sequence[Mapping[str, Any]] = ()) -> list[Observation]:
    """Promote the simulator's own error diagnostics to surfaced observations.

    These are the highest-authority facts — the simulator classified them, we
    only relay. They are also the gap (b) fix: a ``completed`` run with a
    ``singular matrix`` line now surfaces it prominently instead of leaving it
    buried in ``errors`` while the headline says success.

    Each carries the ``severity`` the simulator wrote on its line, read off
    the run's condition that holds it (``said``, the run row's diagnostics),
    or none where no condition names one: a line relayed without its
    condition (a solve-failure line) carries none rather than one the server
    made up.
    """
    severity = {
        line["text"]: line["severity"]
        for c in said
        for line in said_lines(c)
        if isinstance(line.get("severity"), str)
    }

    def with_severity(item: Observation, text: str) -> Observation:
        if text in severity:
            item["severity"] = severity[text]
        return item

    obs: list[Observation] = []
    # An error that names a ``.meas`` is relayed once, as the measurement that
    # failed (below), not again as a log line.
    seen: set[str] = set()
    measured = {
        str(entry.get("raw_block"))
        for entry in summary.get("meas_errors", []) or []
        if isinstance(entry, dict)
    }
    for err in summary.get("errors", []) or []:
        first_line = err.splitlines()[0] if err else err
        if err in measured or first_line in seen:
            continue
        seen.add(first_line)
        obs.append(
            with_severity(
                {
                    "code": "log_error",
                    "kind": "relay",
                    "detail": first_line,
                    "evidence": {"log": err},
                },
                err,
            )
        )
    for entry in summary.get("meas_errors", []) or []:
        directive = entry.get("directive", "") if isinstance(entry, dict) else str(entry)
        obs.append(
            with_severity(
                {
                    "code": "meas_parse_error",
                    "kind": "relay",
                    "detail": f".meas failed to parse: {directive}",
                    "evidence": entry if isinstance(entry, dict) else {"directive": directive},
                },
                str(entry.get("raw_block", "")) if isinstance(entry, dict) else "",
            )
        )
    return obs


#: The codes of what a run's case observed that every read of its results by
#: job repeats (``services.experiment_run_context``): facts about what the
#: raw holds, or how its names are spelled, which a read of one signal would
#: otherwise not see. ``partial_raw`` (a stopped run's kept raw) is added to
#: the read by ``services.partial_raw_observation`` itself.
READ_BY_JOB = frozenset(
    {"subcircuit_nodes_not_saved", "raw_names_disagree", "saves_not_stored"}
)

#: Every other code a run's case can carry, and why a read of the run by job
#: does not repeat it. A case's observations are on the job's receipt
#: (``run_experiments``, ``jobs``); ``tests/test_replies_name_what_they_left_out.py``
#: fails a code a case can carry that is on neither list.
NOT_READ_BY_JOB: Mapping[str, str] = {
    # What the run did, which its receipt says and no read of the raw changes.
    "control_write_injected": "how the run was launched; the raw it wrote is read as written",
    "late_simulator_exit": "when the simulator exited; the raw it left is read as written",
    "kill_unconfirmed": "a run with no confirmed exit leaves no raw a read takes",
    "kill_attempt_failed": "a failed kill attempt; the run's raw, if any, is read as written",
    "partial_progress": "how far a stopped run got; partial_raw says it of the raw a read takes",
    "run_progress": "how far a running case has got, said only while it runs",
    # What the run's log says; a read of the raw does not read the log.
    "logopinfo_withheld": (
        "the log's device table, not the raw: every read of that table by job "
        "gives it with an empty table (services.device_table_unread), and no other read"
    ),
    "native_log_output": "a run that wrote no raw, so no read of a raw by job is made",
    "missing_required_raw": "a run that wrote no raw, so no read of a raw by job is made",
    "ngspice_mode_unverified": (
        "how ngspice read the deck, which the server's deck models use; the raw's traces "
        "are read as the raw spells them"
    ),
    # Where the deck came from; the staged copy is what ran.
    "circuit_id_derived": "how the circuit was named, not what its run wrote",
    "micro_sign_folded": "how the staged deck spells a suffix, not what its run wrote",
    "value_read_otherwise": "how another build reads the source, not what its run wrote",
    "live_include": "which input the run read live, not what its run wrote",
    "live_data_file": "which input the run read live, not what its run wrote",
    "data_path_unrecorded": "how the staged deck names a data file, not what its run wrote",
    "source_unavailable_after_staging": "the source after staging; the staged copy ran",
    "source_modified_after_staging": "the source after staging; the staged copy ran",
}


def saves_not_stored(
    saves: SaveList,
    traces: Sequence[tuple[str, str | None]],
    naming: Naming,
    *,
    circuit: str,
) -> Observation | None:
    """Name each entry of a deck's ``.save`` list (``saves``, from
    ``lint_rules.save_list``, the reading the lint makes before the run) that
    the run's results do not hold, and each this check could not settle; None
    when every entry is held.

    ``traces`` is the raw's first plot (``raw_parser.first_plot_traces``),
    each read as ``naming`` writes a raw (``device_names.trace_key``). An
    entry is not stored where the reader is known to store nothing for it (an
    ngspice ``.save`` of a colon path, another LTspice build's spelling, both
    measured or recorded), or where it is a node voltage whose trace the raw
    lacks. A current or device value the raw does not hold under the ``.save``'s
    own spelling (ngspice writes ``v1#branch`` for ``i(v1)``), an entry whose
    reading is not established, and a pattern (``V(x1:*)``) are not checked:
    which name the run would store is not established, so none is guessed."""
    stored = {trace_key(name, xvii=naming.xvii) for name, _ in traces}
    missing: list[dict[str, str]] = []
    unchecked: list[dict[str, str]] = []
    if naming.family not in ("ltspice", "ngspice"):
        # No recording or live run shows what this simulator's .save stores
        # (Xyce's .SAVE writes an initial-condition file), so nothing is
        # said to be missing.
        reason = (
            f"what {naming.family or 'this simulator'}'s .save stores is not recorded"
        )
        entries = [text for text, _ in saves.entries] + list(saves.patterns)
        if saves.unreadable:
            entries.append(".save")
        if not entries:
            return None
        return Observation(
            code="saves_not_stored",
            kind="coverage",
            detail=f"{circuit}: not checked against the results: .save {', '.join(entries)}.",
            evidence={
                "circuit": circuit,
                "not_checked": [{"save": text, "reason": reason} for text in entries],
            },
        )
    if saves.unreadable:
        unchecked.append({"save": ".save", "reason": "a .save card could not be read"})
    for text, signal in saves.entries:
        if signal.key is not None:
            if signal.key in stored:
                continue
            if signal.key[0] == "v":
                missing.append(
                    {"save": text, "reason": f"the results hold no trace that {text} names"}
                )
            else:
                unchecked.append(
                    {
                        "save": text,
                        "reason": (
                            "which name the run stores for a current or a device value is "
                            "not established, and the results hold none spelled as the "
                            ".save spells it"
                        ),
                    }
                )
        elif not signal.established:
            unchecked.append({"save": text, "reason": signal.why or "not established"})
        elif (
            saves.includes
            and naming.family == "ltspice"
            and signal.spelled is None
            and not signal.voids_save
        ):
            unchecked.append(
                {
                    "save": text,
                    "reason": (
                        "the deck reads other files, whose subcircuit instances this check "
                        f"did not read ({signal.why})"
                    ),
                }
            )
        else:
            missing.append({"save": text, "reason": signal.why or "it stores nothing"})
    for pattern in saves.patterns:
        unchecked.append(
            {"save": pattern, "reason": "which signals a pattern saves is not recorded"}
        )
    if not missing and not unchecked:
        return None
    parts = []
    if missing:
        names = ", ".join(item["save"] for item in missing)
        parts.append(
            f"{circuit}: the run's results hold nothing for .save {names}, so a .meas "
            "or a read of it gives no value."
        )
    if unchecked:
        names = ", ".join(item["save"] for item in unchecked)
        parts.append(f"Not checked against the results: .save {names}.")
    evidence: dict[str, Any] = {"circuit": circuit}
    if missing:
        evidence["not_stored"] = missing
    if unchecked:
        evidence["not_checked"] = unchecked
    return Observation(
        code="saves_not_stored", kind="coverage", detail=" ".join(parts), evidence=evidence
    )


def subcircuit_nodes_not_saved(
    flat: FlatNetlist,
    traces: Sequence[tuple[str, str | None]],
    naming: Naming,
    *,
    saves: bool,
    circuit: str,
) -> Observation | None:
    """Say that a run's raw holds no node inside the deck's subcircuits, or
    None.

    It applies where the deck calls a subcircuit whose body has a net that is
    not a port (``flat``, from ``netlist_graph.flatten_graph``, names such a
    net ``<instance path>/<net>``) and no trace of ``traces`` (the raw's first
    plot, as ``raw_parser.first_plot_traces`` reads it) is a node voltage
    inside any instance it calls, its path read as ``naming`` writes a raw
    (``device_names.trace_key``). A deck whose subcircuits have only ports
    (``raw/op_subckt``) is never said to miss one.

    The causes are the recorded ones: what the deck's ``.save`` cards name
    (``saves``), and, on LTspice, the "Save subcircuit node voltages"
    setting, off by default (``raw/subckt_names_tran`` and its ``_saved``
    recording, on LTspice 26 and XVII). Whether a deck can turn the setting
    on, or a ``.save`` with a wildcard saves one instance's nodes, is not
    recorded, and the detail claims neither."""
    calls = [split_path(call.ref) for call in flat.calls]
    inner: list[tuple[DevicePath, str]] = []
    for path in calls:
        prefix = f"{path_text(path)}/"
        inner.extend(
            (path, net[len(prefix) :]) for net in sorted(flat.nets) if net.startswith(prefix)
        )
    if not inner:
        return None
    instances = set(calls)
    for name, declared in traces:
        if not is_node_voltage(name, declared or None):
            continue
        _, path = trace_key(name, xvii=naming.xvii)
        if len(path) > 1 and path[:-1] in instances:
            return None
    scope, net = inner[0]
    example = f"V({spelled_name(scope, net, naming)})"
    causes = []
    if saves:
        causes.append("the deck's .save cards, which leave out every such node")
    if naming.family == "ltspice":
        causes.append(
            "LTspice's 'Save subcircuit node voltages' setting, off by default, without "
            "which LTspice stores a subcircuit's inner node only where a .save names it "
            "in the build's own spelling"
        )
    detail = (
        f"The raw of {circuit} holds no node voltage inside the deck's subcircuit "
        f"instances, whose bodies have nets of their own ({example} among them), so a "
        "read of such a node finds no trace."
    )
    if causes:
        detail += " What decides it: " + "; and ".join(causes) + "."
    return Observation(
        code="subcircuit_nodes_not_saved",
        kind="coverage",
        detail=detail,
        evidence={
            "circuit": circuit,
            "example": example,
            "instances": len({path for path, _ in inner}),
            "inner_nets": len(inner),
            "saves": saves,
        },
    )


def measurements_failed_in_every_run(
    failed_by_run: Sequence[Collection[str]],
    requested: Collection[str] | None,
    *,
    circuit: str | None = None,
) -> Observation | None:
    """The ``.meas`` that gave no value in any run read, or None for none.

    ``failed_by_run`` holds, for each run whose measurements were read, the
    names it gave no value for at any step (``log_types.given_no_value``:
    printed failed at every step, or not printed at all); ``requested`` the
    names the deck's own text asks for (:func:`parse_requested_outputs`), or
    None where the deck was not read. A name is matched as the log reader
    holds it (``log_types.measurement_named``). The one answer to "which
    measurements did no run give": the analysis and the job receipt both ask
    it, each over the runs it read. When the names cover every requested
    ``.meas``, the detail says so, and that a ``.meas`` in an included file
    is not counted among them."""
    runs = [list(names) for names in failed_by_run]
    if not runs:
        return None
    common = [
        name
        for name in runs[0]
        if all(measurement_named(name, others) is not None for others in runs[1:])
    ]
    if not common:
        return None
    asked = list(requested) if requested is not None else None
    # Named as the deck spells them where it asks for them.
    names = [name for name in asked or () if measurement_named(name, common) is not None]
    names += [name for name in common if measurement_named(name, names) is None]
    every = bool(asked) and all(
        measurement_named(name, common) is not None for name in asked or ()
    )
    where = f" of {circuit}" if circuit is not None else ""
    detail = (
        f".meas {', '.join(names)} gave no value in any of the {len(runs)} run(s){where} "
        "read: the simulator printed "
        + ("it" if len(names) == 1 else "each")
        + " as failed, or printed nothing for it"
        + (
            ". That is every requested .meas the deck's own text holds (one in an "
            "included file is not counted)."
            if every
            else "."
        )
        + " The measurements recipe reads each run's values."
    )
    evidence: dict[str, Any] = {"measurements": names, "runs": len(runs)}
    if asked is not None:
        evidence["requested"] = len(asked)
    if circuit is not None:
        evidence["circuit"] = circuit
    return Observation(
        code="meas_failed_in_every_run",
        kind="reconciliation",
        detail=detail,
        evidence=evidence,
    )


def measurements_not_read_observation(
    reason: str, requested: Sequence[str] | None
) -> Observation:
    """Say that the run's measurements have no value here because they were
    not read, and why (``reason``, from ``services.measurements_unread``, the
    run row's ``measurements_coverage``); ``requested`` is the ``.meas`` the
    deck requests, None where the deck was not read."""
    asked = (
        f"The deck requests {len(requested)} .meas, and none is reported"
        if requested is not None
        else "No .meas result of this run is reported"
    )
    evidence: dict[str, Any] = {"reason": reason}
    if requested is not None:
        evidence["requested"] = list(requested)
    return {
        "code": "measurements_not_read",
        "kind": "coverage",
        "detail": f"{asked}: {MEASUREMENTS_UNREAD[reason]} ({reason}).",
        "evidence": evidence,
    }


def reconciliation_observations(
    summary: dict, requested: dict[str, list[str]], said: Sequence[Mapping[str, Any]] = ()
) -> list[Observation]:
    """Surface requested ``.meas``/``.four`` outputs that were not produced.

    Classifies each miss as ``failed`` (LTspice printed it failed at every
    step — in ``failed_measurements``), ``skipped_in_batch_mode`` (ngspice
    can't evaluate in batch mode — its ``measurement_skipped_batch`` condition
    in ``said``), or ``missing`` (silently absent — the real trust gap). A
    measurement that failed at some steps of a stepped run and was taken at the
    others was produced. The fact, not a verdict: the model decides whether a
    missing measurement matters.
    """
    obs: list[Observation] = []

    # Which requested .meas the run gave no value for, and which of those it
    # printed failed, is run_measurement_facts' answer, the one the run row
    # and the every-run observation read; this only adds why.
    facts = run_measurement_facts(summary, requested.get("meas") or [])
    no_value = given_no_value(facts)
    failed = facts.get("failed") or []
    # Whether ngspice skipped a .meas or .four in batch mode is read off the
    # run's conditions by their codes (``said``), not the warnings list: its
    # lines saying so carry no severity word (log_conditions).
    skipped = {str(c.get("code")) for c in said}
    meas_batch_skipped = "measurement_skipped_batch" in skipped
    four_batch_skipped = "fourier_skipped_batch" in skipped

    for name in requested.get("meas", []):
        if measurement_named(name, no_value) is None:
            continue
        if measurement_named(name, failed) is not None:
            reason = "failed"
        elif meas_batch_skipped:
            reason = "skipped_in_batch_mode"
        else:
            reason = "missing"
        obs.append(
            {
                "code": "unmet_request",
                "kind": "reconciliation",
                "detail": f".meas '{name}' was requested but not produced ({reason})",
                "evidence": {"name": name, "request_kind": "meas", "reason": reason},
            }
        )

    # Fourier is reconciled coarsely: parse_fourier_data does not reliably carry
    # the per-signal name, so we only flag when NO Fourier block was produced at
    # all for a deck that asked for one.
    if requested.get("four") and not summary.get("fourier"):
        reason = "skipped_in_batch_mode" if four_batch_skipped else "missing"
        for signal in requested["four"]:
            obs.append(
                {
                    "code": "unmet_request",
                    "kind": "reconciliation",
                    "detail": f".four '{signal}' was requested but no Fourier data was produced ({reason})",
                    "evidence": {"name": signal, "request_kind": "four", "reason": reason},
                }
            )
    return obs


def value_observations(
    value_traces: dict[str, np.ndarray],
    *,
    source_reference: tuple[str, float] | None = None,
    node_voltages: Collection[str] | None = None,
) -> list[Observation]:
    """Surface salient numbers in the trace data: non-finite samples and
    extreme magnitudes.

    Pure fact-surfacing. ``non_finite`` (NaN/Inf) is unambiguous; ``extreme_value``
    lifts a |value| past the salience threshold into view with a neutral note
    that it is *often* a gmin/floating-node artifact — phrased as something to
    verify, never as a verdict that the run is wrong.

    ``source_reference`` — ``(name, amplitude)`` of the deck's largest
    independent voltage source — arms a third ``extreme_value`` trigger: a
    node voltage whose peak dwarfs every source in the deck (see
    ``_SOURCE_RELATIVE_RATIO``). Passed only for real-valued analyses
    (.tran/.op) where "node voltage vs. drive level" is meaningful.
    ``node_voltages`` names the traces that hold one, as the plot's declared
    types say (``raw_parser.node_voltages``); without it a trace is read by
    its name alone, as ``trace_names.trace_quantity`` reads an untyped trace.
    """
    if node_voltages is None:
        node_voltages = [name for name in value_traces if is_node_voltage(name, None)]
    voltages = frozenset(node_voltages)
    obs: list[Observation] = []
    for name, arr in value_traces.items():
        if arr.size == 0:
            continue
        # Complex AC traces: reason about magnitude.
        mag = np.abs(arr)
        finite = np.isfinite(mag)
        if not finite.all():
            n_bad = int((~finite).sum())
            obs.append(
                {
                    "code": "non_finite",
                    "kind": "value",
                    "detail": f"{name} contains {n_bad} non-finite (NaN/Inf) sample(s) of {arr.size}",
                    "evidence": {"trace": name, "non_finite_count": n_bad, "total": int(arr.size)},
                }
            )
        if finite.any():
            finite_mag = mag[finite]
            peak = float(finite_mag.max())
            flagged_before = len(obs)
            if peak >= _EXTREME_VALUE_SALIENCE:
                obs.append(
                    {
                        "code": "extreme_value",
                        "kind": "value",
                        "detail": (
                            f"{name} reaches |{peak:.3g}| (≥ {_EXTREME_VALUE_SALIENCE:.0g}); "
                            "often a gmin/floating-node artifact — verify the node has a DC "
                            "path to ground"
                        ),
                        "evidence": {"trace": name, "peak_abs": peak},
                    }
                )
            elif peak >= _EXTREME_VALUE_FLOOR:
                # Self-relative blow-up: a peak past the floor that dwarfs the
                # trace's typical level. Median over ALL finite samples (zeros
                # included), not just nonzero — otherwise a node that idles at
                # exactly 0 and then explodes leaves only the spike in the
                # sample set, so center == peak and the divergence hides. A zero
                # median (the trace is 0 for most of the run) means any peak past
                # the floor rose from a flat baseline — itself the signature.
                center = float(np.median(finite_mag))
                zero_baseline = center == 0.0  # ratio → ∞; short-circuits the divide below
                if zero_baseline or peak / center >= _EXTREME_VALUE_RATIO:
                    if zero_baseline:
                        detail = (
                            f"{name} peaks at |{peak:.3g}| from a flat ~0 baseline; often a "
                            "numerical divergence — verify the solve converged"
                        )
                    else:
                        detail = (
                            f"{name} peaks at |{peak:.3g}|, ~{peak / center:.0g}× its typical "
                            f"magnitude (~{center:.3g}); often a numerical divergence — verify "
                            "the solve converged"
                        )
                    obs.append(
                        {
                            "code": "extreme_value",
                            "kind": "value",
                            "detail": detail,
                            "evidence": {"trace": name, "peak_abs": peak, "median_abs": center},
                        }
                    )
            # Source-relative trigger — only when the absolute/self-relative
            # gates stayed quiet (one extreme_value per trace is enough) and
            # only for node voltages, whose scale the V-source reference
            # actually bounds: not a current, nor a device's parameter such
            # as ngspice's v(@m1[vth]).
            if len(obs) == flagged_before and source_reference is not None and name in voltages:
                src_name, src_amp = source_reference
                if peak >= src_amp * _SOURCE_RELATIVE_RATIO:
                    obs.append(
                        {
                            "code": "extreme_value",
                            "kind": "value",
                            "detail": (
                                f"{name} reaches |{peak:.3g}|, ~{peak / src_amp:.0g}× the "
                                f"largest independent voltage source in the deck "
                                f"({src_name} = {src_amp:.3g} V); often a diverging solve "
                                "or an unintended operating point (legitimate for "
                                "transformer/resonant step-up) — verify against the "
                                "intended drive level"
                            ),
                            "evidence": {
                                "trace": name,
                                "peak_abs": peak,
                                "source_name": src_name,
                                "source_amplitude": src_amp,
                            },
                        }
                    )
    return obs


#: What a recording establishes about a ``.meas`` that cannot be taken, by the
#: rules a run's log was read with (``log_conditions.rules_for``), and the
#: recording that shows it. A reading not listed has nothing established.
_FAILED_MEAS_ESTABLISHED: dict[str, str] = {
    "ltspice26": (
        "LTspice 26 refuses a deck whose .meas card does not parse, and a refused "
        "run measures nothing (recorded: log/meas_bad_syntax)"
    ),
}

#: The recording that would settle whether a failed ``.meas`` leaves the
#: others unmeasured where none is established.
_FAILED_MEAS_UNSETTLED = (
    "Whether the other .meas go unmeasured because one failed is not established: "
    "LTspice XVII takes the others in log/meas_bad_syntax, and no recording shows a "
    "build leaving them unmeasured; a recording of a deck with a failing .meas beside "
    "ones that would measure, whose log holds no measurement, would settle it"
)


def _failed_meas_cards(obs: list[Observation], said: Sequence[Mapping[str, Any]]) -> list[str]:
    """The ``.meas`` cards a run's log says failed: those the summary relays as
    ``meas_errors``, and each ``measurement_error`` condition's card where the
    simulator gave it no word (LTspice 26's refusal), in order, once each."""
    cards: list[str] = [
        str(o.get("evidence", {}).get("directive", o.get("detail")))
        for o in obs
        if o.get("code") == "meas_parse_error"
    ]
    for condition in said:
        if condition.get("code") != "measurement_error":
            continue
        texts = [line["text"] for line in said_lines(condition)] or [str(condition["first_text"])]
        cards.extend(
            line.strip()
            for text in texts
            for line in text.splitlines()
            if line.strip().casefold().startswith(".meas")
        )
    return list(dict.fromkeys(cards))


def meas_batch_abort_observation(
    obs: list[Observation],
    *,
    measured: bool = False,
    said: Sequence[Mapping[str, Any]] = (),
    reading: str | None = None,
) -> Observation | None:
    """Link requested ``.meas`` the log does not hold to the ``.meas`` the log
    says failed, saying only what a recording establishes for the build that
    wrote it (``reading``, the rules its log was read with).

    It fires only when the log holds no measurement (``measured`` false): a
    log that holds one shows the ``.meas`` were evaluated, as LTspice XVII's
    log/meas_bad_syntax does around the one it cannot take. A failed
    directive's own miss is explained by its own error and is not counted.
    Correlates across families, so it runs over the assembled list.
    """
    if measured:
        return None
    cards = _failed_meas_cards(obs, said)
    failed = [name for card in cards if (name := extract_meas_name(card))]
    # request_kind gate: .four misses also carry reason="missing", but the
    # Fourier pipeline is unrelated to a failed .meas.
    missing = [
        o
        for o in obs
        if o.get("code") == "unmet_request"
        and o.get("evidence", {}).get("reason") == "missing"
        and o.get("evidence", {}).get("request_kind") == "meas"
        and measurement_named(str(o.get("evidence", {}).get("name", "")), failed) is None
    ]
    if not (cards and missing):
        return None
    established = _FAILED_MEAS_ESTABLISHED.get(reading or "")
    evidence: dict[str, Any] = {
        "missing": [o.get("evidence", {}).get("name") for o in missing],
        "failed_directives": cards,
        "established": established is not None,
    }
    detail = (
        f"{len(missing)} .meas requested were not produced and the log holds no "
        f"measurement, while {len(cards)} .meas directive(s) failed; fix the failing "
        "directive first. "
    )
    if established is not None:
        detail += established + "."
    else:
        detail += _FAILED_MEAS_UNSETTLED + "."
        evidence["settled_by"] = (
            "a recording of a deck with a failing .meas beside ones that would "
            "measure, whose log holds no measurement"
        )
    return {
        "code": "meas_batch_abort",
        "kind": "reconciliation",
        "detail": detail,
        "evidence": evidence,
    }


def surface_observations(
    summary: dict,
    *,
    analysis: Analysis | None = None,
    requested: dict[str, list[str]] | None = None,
    value_traces: dict[str, np.ndarray] | None = None,
    source_amplitudes: dict[str, float] | None = None,
    node_voltages: Collection[str] | None = None,
    measurements_unread: str | None = None,
    said: Sequence[Mapping[str, Any]] = (),
    reading: str | None = None,
) -> list[Observation]:
    """Assemble the full observation list for a result summary.

    ``said`` is the run's conditions with their worded lines, and ``reading``
    the rules its log was read with (``log_conditions.rules_for``).

    ``measurements_unread`` is why the log's measurements were not read
    (``services.measurements_unread``): the requested ``.meas`` are then said
    once to be unread for that reason (``measurements_not_read_observation``),
    never each "missing".

    Passing ``value_traces`` is the decision to surface value facts: a caller
    that loaded them wants them scanned, and one that did not passes None.

    ``source_amplitudes`` (from ``parse_source_amplitudes``) arms the
    source-relative ``extreme_value`` trigger for the selected plot's decoded
    ``analysis`` when that is a transient or an operating point, for the
    traces ``node_voltages`` names (``value_observations``).
    """
    obs: list[Observation] = []
    obs.extend(relay_observations(summary, said))
    if measurements_unread is not None and (
        requested.get("meas")
        if requested is not None
        # A summary of a raw by path has no deck to say what it requests: it
        # still says a simulator's measurements are not read. A log that was
        # not read is said by the summary's diagnostics_coverage.
        else measurements_unread == "simulator_unsupported"
    ):
        asked = None if requested is None else requested["meas"]
        obs.append(measurements_not_read_observation(measurements_unread, asked))
        if requested is not None:
            requested = {**requested, "meas": []}
    if requested:
        obs.extend(reconciliation_observations(summary, requested, said))
        abort = meas_batch_abort_observation(
            obs, measured=bool(summary.get("measurements")), said=said, reading=reading
        )
        if abort is not None:
            obs.append(abort)
    if value_traces is not None:
        source_reference: tuple[str, float] | None = None
        # Only where "node voltage vs. drive level" is meaningful: transient
        # and operating point. AC/noise traces are per-frequency small-signal
        # gains, and a .dc sweep's source range comes from the .dc line, not
        # the V-card this parser read.
        if source_amplitudes and analysis in ("transient", "op"):
            source_reference = max(source_amplitudes.items(), key=lambda kv: kv[1])
        obs.extend(
            value_observations(
                value_traces, source_reference=source_reference, node_voltages=node_voltages
            )
        )
    return obs
