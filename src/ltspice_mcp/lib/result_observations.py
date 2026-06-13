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
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from ltspice_mcp.lib.spice_lex import extract_meas_name

# Salience threshold for lifting a node value into view. The gmin/floating-node
# artifact lands at ~1e30, so 1e8 has enormous headroom; a legitimate HV design
# may also clear it, but surfacing the value is a fact the model reconciles
# against intent, not an accusation — so a loose threshold is fine by design.
_EXTREME_VALUE_SALIENCE = 1e8


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
    evidence: dict


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


def value_observations(value_traces: dict[str, np.ndarray]) -> list[Observation]:
    """Surface salient numbers in the trace data: non-finite samples and
    extreme magnitudes.

    Pure fact-surfacing. ``non_finite`` (NaN/Inf) is unambiguous; ``extreme_value``
    lifts a |value| past the salience threshold into view with a neutral note
    that it is *often* a gmin/floating-node artifact — phrased as something to
    verify, never as a verdict that the run is wrong.
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
            peak = float(mag[finite].max())
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
    return obs


def surface_observations(
    summary: dict,
    *,
    requested: dict[str, list[str]] | None = None,
    value_traces: dict[str, np.ndarray] | None = None,
    value_scan: str = "off",
) -> list[Observation]:
    """Assemble the full observation list for a result summary.

    ``value_scan`` is the caller's explicit coverage decision:
    - ``"scan"``    — ``value_traces`` were loaded; scan them.
    - ``"skipped_large"`` — traces were not loaded (bounded success path);
      surface a coverage observation so the gap is visible.
    - ``"off"``     — value surfacing not applicable for this caller.
    """
    obs: list[Observation] = []
    obs.extend(relay_observations(summary))
    if requested:
        obs.extend(reconciliation_observations(summary, requested))
    if value_scan == "scan" and value_traces is not None:
        obs.extend(value_observations(value_traces))
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
