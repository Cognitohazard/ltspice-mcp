"""Result observation surfacing — a "surfacer", deliberately not a "judger".

The consumer of a simulation result here is an LLM agent with its own physics
knowledge: it already knows 1e30 V is a floating node. So this layer does NOT
render a trust verdict (a ``confidence: degraded`` label would inject our
subjective judgment in place of the model's better-informed one, and a wrong
label is a false accusation the model may parrot or learn to ignore). Instead it
*surfaces facts* the model would otherwise miss or have to dig for, and lets the
model judge.

This module is the canonical implementation of the repo-wide "Result-trust
doctrine — surface, don't judge" in CLAUDE.md. Design rules:
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
  doctrine-governed by this module.
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
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.spice_lex import extract_meas_name

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
# lands in. Doctrine: a relative signature, surfaced as a fact, not a verdict.
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
# surfaced, which the model dismisses cheaply in context (doctrine above).
_SOURCE_RELATIVE_RATIO = 20.0  # peak |V| ≥ this many × the largest V-source amplitude

# Transient-function keywords a V-card may carry; used by the best-effort
# source-amplitude parser below.
_SOURCE_FUNCTION_KEYWORDS = frozenset({"sin", "sine", "pulse", "exp", "pwl", "sffm", "am"})


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


def parse_requested_outputs(netlist_text: str) -> dict[str, list[str]]:
    """Extract requested ``.meas``/``.four`` names from a netlist deck.

    Returns ``{"meas": [name, ...], "four": [signal, ...]}`` preserving the
    original-case tokens. The ``.meas`` name is pulled via the canonical
    ``spice_lex.extract_meas_name`` (the same extractor the lexer/validator use)
    so reconciliation matches how produced names are derived elsewhere, instead
    of re-implementing the analysis-keyword skip. Best-effort — reconciliation is
    advisory.
    """
    meas: list[str] = []
    four: list[str] = []
    for line in netlist_text.splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        head = tokens[0].lower()
        if head.startswith(".meas"):
            name = extract_meas_name(line)
            if name:
                meas.append(name)
        elif head in (".four", ".fourier"):
            # ``.four <freq> <signal> [signal ...]`` — signals start at token 2.
            four.extend(tokens[2:])
    return {"meas": meas, "four": four}


def _parse_source_number(tok: str) -> float:
    """``parse_spice_value`` plus the bare voltage-unit form LTspice accepts.

    ``5V`` has no scale suffix, only a unit tail, so ``parse_spice_value``
    raises on it — but it is a valid V-card level. In this voltage-source
    context a single trailing ``v``/``V`` is safely a unit, so retry without it
    (``1mV`` etc. already parse via the scale-suffix path).
    """
    try:
        return parse_spice_value(tok)
    except ValueError:
        if len(tok) > 1 and tok[-1] in "vV":
            return parse_spice_value(tok[:-1])
        raise


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
                    fn_args.append(_parse_source_number(arg))
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
                    _parse_source_number(spec[i])
                except ValueError:
                    break
                i += 1
            continue
        try:
            v = _parse_source_number(tok)
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
_EQUALS_WS_RE = re.compile(r"\s*=\s*")


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
    PWL, SFFM and AM specs, ``+`` continuation lines, bare ``5V`` unit tails,
    and simple ``{name}`` references to literal ``.param`` constants.

    All-or-nothing: if ANY V-card cannot be bounded (behavioral, unresolvable
    ``{expr}``, a spec cut short), the whole result is ``{}`` and the trigger
    stays disarmed — the dropped card might have been the deck's largest
    source, and a max() over the remainder would state a false "largest
    independent voltage source" fact. Quieter, never wrong.
    """
    cards: list[str] = []
    for line in netlist_text.splitlines():
        # Drop inline ``; comment`` trails — a commented V-card would otherwise
        # read as unparseable and falsely disarm the whole deck.
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("+") and cards:
            cards[-1] += " " + stripped[1:]
        else:
            cards.append(stripped)

    # Literal .param constants, for resolving {name} source values — the
    # common parameterized-rail idiom (.param vdd=12 / V1 vdd 0 {vdd}).
    # Non-numeric params (expressions) are simply absent from the map.
    params: dict[str, float] = {}
    for card in cards[1:]:
        if card.lower().startswith(".param"):
            for name, value in _PARAM_ASSIGN_RE.findall(card):
                try:
                    params[name.lower()] = parse_spice_value(value)
                except ValueError:
                    continue

    out: dict[str, float] = {}
    # cards[0] is the deck title line, never an element card.
    for card in cards[1:]:
        if not card or card[0] not in "Vv":
            continue
        # Collapse spaces around '=' so ``Rser = 1`` tokenizes as one
        # key=value token instead of a bare word that reads as unparseable.
        card = _EQUALS_WS_RE.sub("=", card)
        tokens = card.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if len(tokens) < 4:
            continue
        amp = _v_card_amplitude([_resolve_param_ref(tok, params) for tok in tokens[3:]])
        if amp is None or not math.isfinite(amp):
            return {}  # an unbounded V-card poisons the max — disarm entirely
        if amp > 0:
            out[tokens[0]] = amp
    return out


def deck_observation_inputs(
    netlist: Path,
) -> tuple[dict[str, list[str]] | None, dict[str, float] | None]:
    """Read a deck once and parse both observation inputs derived from it:
    the requested ``.meas``/``.four`` names (reconciliation) and the
    independent voltage-source amplitudes (source-relative trigger).

    The one call every summary path shares, so run-completion and later
    re-inspection feed the surfacer identically. ``(None, None)`` when the
    deck can't be read.
    """
    try:
        text = read_spice_text(netlist)
    except OSError:
        return None, None
    return parse_requested_outputs(text), parse_source_amplitudes(text)


def relay_observations(summary: dict) -> list[Observation]:
    """Promote the simulator's own error diagnostics to surfaced observations.

    These are the highest-authority facts — the simulator classified them, we
    only relay. They are also the gap (b) fix: a ``completed`` run with a
    ``singular matrix`` line now surfaces it prominently instead of leaving it
    buried in ``errors`` while the headline says success.
    """
    obs: list[Observation] = []
    seen: set[str] = set()
    for err in summary.get("errors", []) or []:
        first_line = err.splitlines()[0] if err else err
        if first_line in seen:
            continue
        seen.add(first_line)
        obs.append(
            {
                "code": "log_error",
                "kind": "relay",
                "severity": "error",
                "detail": first_line,
                "evidence": {"log": err},
            }
        )
    for entry in summary.get("meas_errors", []) or []:
        directive = entry.get("directive", "") if isinstance(entry, dict) else str(entry)
        obs.append(
            {
                "code": "meas_parse_error",
                "kind": "relay",
                "severity": "error",
                "detail": f".meas failed to parse: {directive}",
                "evidence": entry if isinstance(entry, dict) else {"directive": directive},
            }
        )
    return obs


def reconciliation_observations(
    summary: dict, requested: dict[str, list[str]]
) -> list[Observation]:
    """Surface requested ``.meas``/``.four`` outputs that were not produced.

    Classifies each miss as ``failed`` (ran but didn't trigger — already in
    ``failed_measurements``), ``skipped_in_batch_mode`` (ngspice can't evaluate
    in batch mode — a recoverable relay signal already in ``warnings``), or
    ``missing`` (silently absent — the real trust gap). The fact, not a verdict:
    the model decides whether a missing measurement matters.
    """
    obs: list[Observation] = []

    produced_meas = {m.lower() for m in (summary.get("measurements") or {})}
    failed = {f.lower() for f in (summary.get("failed_measurements") or [])}
    # A FAILED .meas still appears as a null-valued key in ``measurements`` (and
    # in ``failed_measurements``). Without removing it here, the produced-check
    # below would treat it as produced and skip it — shadowing the "failed"
    # classification, so a failed measurement never surfaced as an observation.
    produced_meas -= failed
    warns = " ".join(summary.get("warnings") or []).lower()
    meas_batch_skipped = "batch mode" in warns and "meas" in warns
    four_batch_skipped = ".fourier line ignored" in warns or "skips .four" in warns

    for name in requested.get("meas", []):
        if name.lower() in produced_meas:
            continue
        if name.lower() in failed:
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
) -> list[Observation]:
    """Surface salient numbers in the trace data: non-finite samples and
    extreme magnitudes.

    Pure fact-surfacing. ``non_finite`` (NaN/Inf) is unambiguous; ``extreme_value``
    lifts a |value| past the salience threshold into view with a neutral note
    that it is *often* a gmin/floating-node artifact — phrased as something to
    verify, never as a verdict that the run is wrong.

    ``source_reference`` — ``(name, amplitude)`` of the deck's largest
    independent voltage source — arms a third ``extreme_value`` trigger: a
    voltage trace whose peak dwarfs every source in the deck (see
    ``_SOURCE_RELATIVE_RATIO``). Passed only for real-valued analyses
    (.tran/.op) where "node voltage vs. drive level" is meaningful.
    """
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
            # only for voltage traces, whose scale the V-source reference
            # actually bounds.
            if (
                len(obs) == flagged_before
                and source_reference is not None
                and name.lower().startswith("v(")
            ):
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


def meas_batch_abort_observation(obs: list[Observation]) -> Observation | None:
    """Causal link between a ``.meas`` parse error and unexplained misses.

    One erroring ``.meas`` directive makes LTspice abandon the whole ``.meas``
    batch — even directives that precede it come back missing. When the
    assembled list holds both a relay ``meas_parse_error`` and reconciliation
    ``missing`` misses, link them so the model fixes the one failing directive
    instead of chasing N phantom misses. Correlates across families, so it runs
    over the assembled list, not inside either family builder.
    """
    parse_errors = [o for o in obs if o.get("code") == "meas_parse_error"]
    # request_kind gate: .four misses also carry reason="missing", but the
    # Fourier pipeline is unrelated to the .meas batch abort — folding them
    # in would state a wrong count and attribution.
    missing = [
        o
        for o in obs
        if o.get("code") == "unmet_request"
        and o.get("evidence", {}).get("reason") == "missing"
        and o.get("evidence", {}).get("request_kind") == "meas"
    ]
    if not (parse_errors and missing):
        return None
    return {
        "code": "meas_batch_abort",
        "kind": "reconciliation",
        "detail": (
            f"{len(missing)} .meas requested but not produced while "
            f"{len(parse_errors)} .meas directive(s) failed to parse — LTspice "
            "skips the remaining .meas evaluation (including directives that "
            "come earlier in the deck) after one bad directive; fix the "
            "failing directive first."
        ),
        "evidence": {
            "missing": [o.get("evidence", {}).get("name") for o in missing],
            "failed_directives": [
                o.get("evidence", {}).get("directive", o.get("detail"))
                for o in parse_errors
            ],
        },
    }


def surface_observations(
    summary: dict,
    *,
    requested: dict[str, list[str]] | None = None,
    value_traces: dict[str, np.ndarray] | None = None,
    value_scan: str = "off",
    source_amplitudes: dict[str, float] | None = None,
) -> list[Observation]:
    """Assemble the full observation list for a result summary.

    ``value_scan`` is the caller's explicit coverage decision:
    - ``"scan"``    — ``value_traces`` were loaded; scan them.
    - ``"skipped_large"`` — traces were not loaded (bounded success path);
      surface a coverage observation so the gap is visible.
    - ``"off"``     — value surfacing not applicable for this caller.

    ``source_amplitudes`` (from ``parse_source_amplitudes``) arms the
    source-relative ``extreme_value`` trigger for real-valued analyses.
    """
    obs: list[Observation] = []
    obs.extend(relay_observations(summary))
    if requested:
        obs.extend(reconciliation_observations(summary, requested))
        abort = meas_batch_abort_observation(obs)
        if abort is not None:
            obs.append(abort)
    if value_scan == "scan" and value_traces is not None:
        source_reference: tuple[str, float] | None = None
        if source_amplitudes:
            # Only where "node voltage vs. drive level" is meaningful: transient
            # and operating point. AC/noise traces are per-frequency small-signal
            # gains, and a .dc sweep's source range comes from the .dc line, not
            # the V-card this parser read.
            sim_type = str(summary.get("sim_type", "")).strip().lower()
            if sim_type.startswith("transient") or sim_type.startswith("operating"):
                source_reference = max(source_amplitudes.items(), key=lambda kv: kv[1])
        obs.extend(value_observations(value_traces, source_reference=source_reference))
    elif value_scan == "skipped_large":
        obs.append(
            {
                "code": "value_scan_skipped",
                "kind": "coverage",
                "detail": (
                    "Result is too large to scan (trace samples exceed the value-scan "
                    "budget); traces were not scanned for NaN/Inf or extreme values. "
                    "Inspect specific signals with signal_stats/query_value if a "
                    "degenerate result is suspected."
                ),
                "evidence": {"point_count": summary.get("point_count")},
            }
        )
    return obs
