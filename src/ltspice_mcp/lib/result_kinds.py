"""What a results plot holds, and where each of its facts is kept.

One place says what a plot is: its analysis, read off its name and first
variable, the axis it is sampled on, and, for a stepped LTspice plot, which
kind of stepped result it is. A kind says how its plot splits into steps, and
whether its axis and its step values are in the raw, in the log or only in
the deck. The decoder splits steps by it and the readers ask it, rather than
each restating a rule. Every kind names the LTspice recordings that establish
it, by case id (``tests/test_result_kinds.py`` holds it to them);
``raw_header`` keeps the byte layouts that were validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.lib.raw_header import RawPlotHeader, leading_parameters

Analysis = Literal[
    "op",
    "transient",
    "ac",
    "fra",
    "dc",
    "noise",
    "tf",
    "pz",
    "sens_dc",
    "sens_ac",
    "disto",
    "unknown",
]
#: The analyses whose results are complex values over frequency: an .AC sweep,
#: and the results file of an LTspice frequency response analysis, which is
#: written in the same layout.
AC_SHAPED: frozenset[Analysis] = frozenset({"ac", "fra"})

_ANALYSIS_NAMES: dict[str, Analysis] = {
    "operating point": "op",
    "transfer function": "tf",
    "pole-zero analysis": "pz",
    "transient analysis": "transient",
    "ac analysis": "ac",
    "frequency response analysis": "fra",
    "dc transfer characteristic": "dc",
    "integrated noise": "noise",
}


def plot_analysis(plot_name: str, first_type: str) -> Analysis:
    """The analysis a plot holds, from its name and its first variable's
    declared type: the one reading of a plot's kind, header in hand or not."""
    label = plot_name.strip().lower()
    if label == "sensitivity analysis":
        return "sens_ac" if first_type.lower() in ("frequency", "hertz") else "sens_dc"
    if label.startswith("distortion -"):
        return "disto"
    if label.startswith("noise spectral density"):
        return "noise"
    return _ANALYSIS_NAMES.get(label, "unknown")


def header_analysis(header: RawPlotHeader) -> Analysis:
    """The analysis a decoded header's plot holds (:func:`plot_analysis`)."""
    return plot_analysis(header.plot_name, header.first_type)


def sampled_axis(analysis: Analysis, first_type: str) -> tuple[str, str] | None:
    """The quantity and kind of a plot's sampled axis, from its analysis and
    its first variable's declared type; None for a plot with no sampled axis,
    such as an operating point or a stepped one-frequency .ac, whose first
    variable is a step's parameter."""
    declared = first_type.lower()
    if analysis == "transient" and declared == "time":
        return "time", "deck_time"
    if (analysis in AC_SHAPED or analysis in ("sens_ac", "disto", "noise")) and declared in (
        "frequency",
        "hertz",
    ):
        return "frequency", "swept_f1" if analysis == "disto" else "frequency"
    if analysis == "dc":
        return "swept_variable", "sweep"
    return None


#: Where a fact of a result is kept.
Source = Literal["raw", "log", "deck"]
#: How a stepped plot holds its steps: each step's sweep in turn, the axis
#: starting over at each; one point a step, the stepped values first; or one
#: point that does not say which step it is.
Stepping = Literal["sweeps_in_turn", "point_a_step", "point_of_no_step"]


#: Where each stepping keeps a step's stepped values: the log's ``.step``
#: lines, the variables the plot opens with, or nowhere.
_STEP_VALUES: dict[Stepping, Source | None] = {
    "sweeps_in_turn": "log",
    "point_a_step": "raw",
    "point_of_no_step": None,
}


@dataclass(frozen=True)
class SteppedKind:
    """One kind of stepped LTspice result.

    ``axis`` is where the sweep's variable is kept, None for a kind with none.
    ``recorded`` names the recordings that establish the kind, a case id for
    every build that recorded it, or ``<build>:<case id>`` for one build.
    """

    name: str
    analyses: frozenset[Analysis]
    stepping: Stepping
    axis: Source | None
    recorded: tuple[str, ...]

    @property
    def step_values(self) -> Source | None:
        """Where each step's stepped values are kept, None where nothing says."""
        return _STEP_VALUES[self.stepping]


#: A stepped .tran, .ac, .dc or frequency response analysis: each step's
#: sweep in turn. A stepped .noise sweep is read the same way, but none is
#: recorded.
STEPPED_SWEEP = SteppedKind(
    "stepped_sweep",
    frozenset({"transient", "ac", "dc", "noise", "fra"}),
    "sweeps_in_turn",
    "raw",
    ("raw/step_tran", "raw/step_temp", "raw/step_ac", "raw/step_dc", "raw/fra_step.fra_1"),
)
#: A stepped .op, over parameters or temperature.
STEPPED_OP = SteppedKind(
    "stepped_op",
    frozenset({"op"}),
    "point_a_step",
    None,
    ("raw/step_op", "raw/step_temp_op", "ltspice17:raw/step_tran_op_raw.op"),
)
#: A stepped .tf.
STEPPED_TF = SteppedKind("stepped_tf", frozenset({"tf"}), "point_a_step", None, ("raw/tf_step",))
#: A stepped .ac at one frequency, which the raw does not store.
STEPPED_AC_ONE_FREQUENCY = SteppedKind(
    "stepped_ac_one_frequency",
    frozenset({"ac"}),
    "point_a_step",
    "deck",
    ("raw/step_ac_one_freq", "raw/step_ac_one_freq_numdgt7"),
)
#: A stepped .noise at one frequency, which the raw does not store.
STEPPED_NOISE_ONE_FREQUENCY = SteppedKind(
    "stepped_noise_one_frequency",
    frozenset({"noise"}),
    "point_a_step",
    "deck",
    ("raw/step_noise_one_freq",),
)
#: The .op.raw of a stepped .tran or .ac on LTspice 26: one point, and
#: nothing says which step it is.
STEPPED_RUN_OP = SteppedKind(
    "stepped_run_op",
    frozenset({"op"}),
    "point_of_no_step",
    None,
    ("ltspice26:raw/step_tran_op_raw.op", "ltspice26:raw/step_ac_op_raw.op"),
)
STEPPED_KINDS: tuple[SteppedKind, ...] = (
    STEPPED_SWEEP,
    STEPPED_OP,
    STEPPED_TF,
    STEPPED_AC_ONE_FREQUENCY,
    STEPPED_NOISE_ONE_FREQUENCY,
    STEPPED_RUN_OP,
)
_BY_NAME = {kind.name: kind for kind in STEPPED_KINDS}


def kind_named(name: str | None) -> SteppedKind | None:
    """The kind a plot descriptor's ``stepped_kind`` names."""
    return None if name is None else _BY_NAME[name]


def stepped_kind(header: RawPlotHeader) -> SteppedKind | None:
    """The kind of stepped LTspice result a plot is; None for an unstepped
    plot, another writer's, or a stepped one no kind describes."""
    if header.dialect != "ltspice" or not header.stepped:
        return None
    analysis = header_analysis(header)
    if leading_parameters(header):
        return next(
            (
                kind
                for kind in STEPPED_KINDS
                if kind.stepping == "point_a_step" and analysis in kind.analyses
            ),
            None,
        )
    if sampled_axis(analysis, header.first_type) is not None:
        return STEPPED_SWEEP if analysis in STEPPED_SWEEP.analyses else None
    if analysis in STEPPED_RUN_OP.analyses and header.point_count == 1:
        return STEPPED_RUN_OP
    return None
