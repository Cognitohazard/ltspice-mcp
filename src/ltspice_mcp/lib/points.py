"""A point: one simulated circuit, and how a reply names it.

A point is a case of a job and, where its deck has ``.step``, one step of that
case's results. A case is the unit of files, status and failure; a step is a
part of one case's results. Every value or trace a reply shows names its
point: ``analyze_results`` by a row's identity fields (:class:`RowIdentity`,
filled by :func:`point_identity`), ``plot_waveform`` by the same fields on each
trace summary and by a trace's legend label (:func:`point_label`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ltspice_mcp.lib.services import AnalysisSource


@dataclass(frozen=True)
class RowIdentity:
    """What identifies one point (one run at one step), in emission order.

    Declared here rather than as a list of key strings so the row shape, the
    attribution block, the projector's alphabet and a plotted trace's summary
    all read off one definition.
    """

    case_id: str | None = None
    run_index: int | None = 0
    step_index: int | None = None
    step_values: dict[str, Any] = field(default_factory=dict)
    assignments: dict[str, Any] = field(default_factory=dict)
    circuit: str | None = None
    deck_sha256: str | None = None
    plot_index: int | None = None
    dialect: str | None = None
    snapshot_id: str | None = None

    def wire(self, keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: getattr(self, key) for key in keys}


def point_identity(
    source: AnalysisSource,
    step: int | None,
    values: dict[str, Any],
    *,
    plot_index: int | None,
) -> RowIdentity:
    """The identity of one step of one resolved run."""
    base = dict(source.identity or {})
    return RowIdentity(
        case_id=base.get("case_id"),
        run_index=base.get("run_index", 0),
        step_index=step,
        step_values=values,
        assignments=dict(base.get("assignments") or {}),
        circuit=base.get("circuit"),
        deck_sha256=base.get("deck_sha256"),
        plot_index=plot_index,
        dialect=base.get("dialect"),
        snapshot_id=base.get("snapshot_id"),
    )


def values_text(values: Mapping[str, Any]) -> str:
    """``name=value`` for each of ``values``, joined by ``;``: a number in its
    shortest general form (``1e-07``), anything else as it is. How a label
    and a CSV's ``step_value`` column write a step's values or a case's
    assignments."""
    return ";".join(
        f"{key}={value:g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in values.items()
    )


#: Most assignments a label names. Past this many that differ among the cases
#: drawn (a Monte Carlo case draws a value for each varied part), a legend
#: label would be a line of numbers, so the label names the case by its id;
#: the reply's list of cases drawn gives every assignment.
LABEL_MAX_ASSIGNMENTS = 3


def assignments_that_differ(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """The names of the assignments whose values are not the same in every one
    of ``cases`` (each a case's assignments), in the order each first appears.
    A name that one case assigns and another does not differs."""
    names: dict[str, None] = {}
    for assignments in cases:
        names.update(dict.fromkeys(assignments))
    missing = object()
    return [
        name
        for name in names
        if len({repr(assignments.get(name, missing)) for assignments in cases}) > 1
    ]


def label_assignments(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any] | None]:
    """For each of ``cases`` (the assignments of each case drawn, in order),
    the assignments its label names: those that differ among the cases, or
    None where they do not tell the case from every other one (nothing
    differs, two cases share them, or more than ``LABEL_MAX_ASSIGNMENTS``
    differ), and its label names its case id instead."""
    differ = assignments_that_differ(cases)
    if not differ or len(differ) > LABEL_MAX_ASSIGNMENTS:
        return [None for _ in cases]
    named = [
        {name: assignments[name] for name in differ if name in assignments}
        for assignments in cases
    ]
    keys = [repr(sorted(subset.items())) for subset in named]
    return [
        subset if keys.count(key) == 1 else None for subset, key in zip(named, keys, strict=True)
    ]


def point_label(
    signal: str,
    *,
    case_id: str | None = None,
    assignments: Mapping[str, Any] | None = None,
    step_values: Mapping[str, Any] | None = None,
    step_index: int | None = None,
) -> str:
    """A trace's legend label: the signal, then in brackets what tells its
    point from the others drawn: the case id where its assignments do not
    (``case_id``), else the assignments that differ, then the step's values,
    or ``step N`` for a step whose values are not known (``step_index``).
    Pass only what tells the points drawn apart; with nothing, the label is
    the signal alone. The one writer of a point's label.

    ``V(out) [R1=1000;r=22]``, ``V(out) [step 2]``, ``V(out) [case-0003]``.
    """
    parts: list[str] = []
    if case_id is not None:
        parts.append(case_id)
    if assignments:
        parts.append(values_text(assignments))
    if step_values:
        parts.append(values_text(step_values))
    elif step_index is not None:
        parts.append(f"step {step_index}")
    return f"{signal} [{';'.join(parts)}]" if parts else signal
