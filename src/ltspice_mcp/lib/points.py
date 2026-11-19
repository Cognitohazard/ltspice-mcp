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

from ltspice_mcp.lib.reply_cuts import CUT, NEVER

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_types import ExperimentCase
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
    """The names of the assignments whose values a label would write
    differently in some of ``cases`` (each a case's assignments), in the
    order each first appears. A name that one case assigns and another does
    not differs. Values are compared as :func:`values_text` writes them, so
    two that a label would print alike (``1000`` and ``1000.0``) do not."""
    names: dict[str, None] = {}
    for assignments in cases:
        names.update(dict.fromkeys(assignments))
    return [
        name
        for name in names
        if len(
            {
                values_text({name: assignments[name]}) if name in assignments else None
                for assignments in cases
            }
        )
        > 1
    ]


def label_assignments(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any] | None]:
    """For each of ``cases`` (the assignments of each case drawn, in order),
    the assignments its label names: those that differ among the cases, or
    None where they do not tell the case from every other one as a label
    writes them (nothing differs, the case assigns none of them, two cases
    would read alike, or more than ``LABEL_MAX_ASSIGNMENTS`` differ), and its
    label names its case id instead."""
    differ = assignments_that_differ(cases)
    if not differ or len(differ) > LABEL_MAX_ASSIGNMENTS:
        return [None for _ in cases]
    named = [
        {name: assignments[name] for name in differ if name in assignments}
        for assignments in cases
    ]
    shown = [values_text(subset) for subset in named]
    return [
        subset if subset and shown.count(text) == 1 else None
        for subset, text in zip(named, shown, strict=True)
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


#: What a reply says of a case with a raw whose record holds no steps: one
#: recorded before the server kept them, one whose run was recovered after its
#: owner stopped, one stopped with its partial raw kept, or one whose run's
#: end-of-run checks have not finished.
STEPS_NOT_RECORDED = (
    "How this case's results are stepped was not recorded when its run ended (a "
    "case recorded before the server kept it, recovered after its server stopped, "
    "or stopped with its partial raw kept), so it is not established here. A read "
    "of its results gives their steps: inspect(kind='results', view='plots') or "
    "plot_waveform with its case_id."
)

#: The ``steps`` a run row and a chart's ``cases[]`` carry.
STEPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 2, "description": "How many steps it holds."},
        "parameters": {
            "type": "array",
            "items": {"type": "string"},
            # One name per .step parameter of the deck: the deck bounds it.
            CUT: NEVER,
            "description": "The stepped parameters, in the order its steps first name them.",
        },
        "not_established": {
            "type": "string",
            "description": (
                "Why how many steps it holds, or their values, were not established; "
                "without count, a number of steps is not known."
            ),
        },
    },
    "additionalProperties": False,
    "description": (
        "Present for a case whose results are stepped, or whose stepping was not "
        "established: how, as decided once when its run ended. A case whose results "
        "are not stepped has none."
    ),
}


def steps_shown(case: ExperimentCase) -> dict[str, Any]:
    """``{"steps": ...}`` for a case whose results are stepped or whose
    stepping is not established, else nothing: what a run row and a chart's
    ``cases[]`` say, read off the case's record (``ExperimentCase.steps``)."""
    steps = case.steps
    if steps is None:
        return {"steps": {"not_established": STEPS_NOT_RECORDED}} if case.raw_file else {}
    if not steps.holds_steps:
        return {}
    if steps.count is None:
        return {"steps": {"not_established": steps.not_established or STEPS_NOT_RECORDED}}
    return {
        "steps": {
            "count": steps.count,
            "parameters": list(steps.parameters),
            **({"not_established": steps.not_established} if steps.not_established else {}),
        }
    }
