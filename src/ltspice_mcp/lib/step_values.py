"""A ``.step`` run's step values, and which step a requested value names.

Every reader that picks a step by its values asks this module: the step
selector behind ``analyze_results``', ``run_experiments.analyze``'s and
``plot_waveform``'s ``step`` (``recipes.StepSelector``), ``RawResult.compare``
pairing a step with another run's, and a decoded raw's ``get_steps`` filter.
It imports nothing heavier than :mod:`ltspice_mcp.lib.format`, so the raw
decoder can ask it too.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from ltspice_mcp.lib.format import format_spice_value, parse_spice_value, same_value
from ltspice_mcp.lib.raw_header import STEPPED_TEMPERATURE

#: How many significant digits a step value is matched to. A stepped .tran,
#: .ac, .dc, .noise or .fra keeps its step values only in its log's ``.step``
#: lines (``result_kinds``). Both builds write an exponent there as C's ``%g``
#: does (``c=1e-007`` on XVII, ``c=1e-07`` on LTspice 26,
#: ``log/step_two_params``), and ``%g`` keeps six significant digits, so
#: ``.step param r list 2.1544346900318838k`` would be read back as 2154.43,
#: two parts in a million from the deck's value. No recording yet has a step
#: value that needs more than six digits: ``raw/step_digits`` is the deck that
#: records how many each build prints. A raw that stores one point a step
#: holds its step values to seven digits or more (its first variable in eight
#: bytes, any other in four).
#:
#: Six digits are coarser than ``format.SAME_VALUE_RTOL``, by a share of the
#: value that depends on its leading digit (five parts in a million for
#: ``1.00000``, half a part in a million for ``9.99999``), so a step value is
#: matched to its digits (:func:`same_step_value`) rather than within any one
#: share. Matching to six refuses no value a source holds to six or more, and
#: joins two values that differ only past the sixth digit, as a raw can hold
#: them; a selector then reads the first of their steps.
STEP_VALUE_DIGITS = 6

#: A value written halfway between two printed ones, as the deck's value to
#: seven digits can be (``2154.435``), lands a hair either side of the
#: midpoint once read in binary. It names the step on either side.
_MIDPOINT_SLACK = 1 + 1e-6


def printed_half_unit(number: float) -> float:
    """How far a step value read as ``number`` can be from the value it was
    printed for: half a unit in the last of its :data:`STEP_VALUE_DIGITS`
    significant digits (0.005 for ``2154.43``). Zero for zero, which ``%g``
    prints for nothing else."""
    if number == 0:
        return 0.0
    exponent = int(f"{number:.{STEP_VALUE_DIGITS - 1}e}".rpartition("e")[2])
    return 0.5 * 10.0 ** (exponent - STEP_VALUE_DIGITS + 1)


def step_axis_key(axis: str, values: Collection[str]) -> str:
    """The key of one step's ``values`` (its mapping, or its names) that a
    caller's step-axis name reads.

    The name itself where a step holds it, else the one that differs from it
    only in case: SPICE compares names without case, and both LTspice builds
    print a parameter the deck steps as ``Rv`` as ``.step rv=...`` in the log
    (recorded: ``log/meas_when_threshold_tran_step``). A stepped temperature
    is held as the deck's ``.step temp`` names it; LTspice's raw of a stepped
    operating point calls the same values ``temperature``, so that name reads
    them too.
    """
    if axis in values:
        return axis
    folded = axis.casefold()
    if folded == "temperature" and STEPPED_TEMPERATURE in values:
        return STEPPED_TEMPERATURE
    return next((key for key in values if key.casefold() == folded), axis)


class StepValues(dict[str, Any]):
    """One step's parameters as the Python API gives them: a ``dict`` keyed as
    the run wrote them, in which a name is found as :func:`step_axis_key`
    reads it, so ``steps[0]["Rv"]`` reads the log's ``rv``."""

    # The rule reads the dict's own keys view, whose lookups are the dict's,
    # not the ones below.
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and step_axis_key(name, dict.keys(self)) in dict.keys(self)

    def __getitem__(self, name: str) -> Any:
        return dict.__getitem__(self, step_axis_key(name, dict.keys(self)))

    def get(self, name: str, default: Any = None, /) -> Any:
        return dict.get(self, step_axis_key(name, dict.keys(self)), default)


def step_number(value: Any) -> float | None:
    """A step value, or a requested one, as the number SPICE reads it; None
    for one that is not a number (``'2k'`` is 2000, ``'fast'`` is None)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = parse_spice_value(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def same_step_value(a: Any, b: Any) -> bool:
    """Whether two step values, or a step's and a requested one, are one value.

    Two numbers are when they are no further apart than the printing of the
    smaller in magnitude moves a value (:func:`printed_half_unit`), however
    each is written: ``'1k'``, ``1e3`` and ``1000.0`` are one value, and
    ``'6.8k'`` and 4000 are not. So the deck's 2154.4346900318838 is the
    ``2154.43`` its log printed for it, and two values a log prints apart,
    ``100.001`` and ``100.002``, are two. Taking the smaller's digits makes
    ``9.999996`` the ``10`` it is printed as, and ``9.99999`` not. A value
    that is not a number is only the same text.

    Not ``format.same_value``, whose part in a million is finer than six
    printed digits keep, and which joins values a log prints apart at any
    share wide enough for those digits (:data:`STEP_VALUE_DIGITS`).
    """
    number_a, number_b = step_number(a), step_number(b)
    if number_a is None and number_b is None:
        return str(a) == str(b)
    if number_a is None or number_b is None:
        return False
    smaller = min(abs(number_a), abs(number_b))
    return abs(number_a - number_b) <= printed_half_unit(smaller) * _MIDPOINT_SLACK


def _same_held_value(a: Any, b: Any) -> bool:
    """Whether a step holds a requested value itself: the same number to the
    shared tolerance (``format.same_value``), or the same text."""
    number_a, number_b = step_number(a), step_number(b)
    if number_a is None and number_b is None:
        return str(a) == str(b)
    if number_a is None or number_b is None:
        return False
    return same_value(number_a, number_b)


def steps_with(step_values: Sequence[Mapping[str, Any]], wanted: Mapping[str, Any]) -> list[int]:
    """The indices of the steps that have every value of ``wanted``, each
    name found as :func:`step_axis_key` finds it. Every step when ``wanted``
    is empty.

    A step that holds a value itself (``format.same_value``) is the one it
    names: a raw holds a stepped operating point's values to full precision,
    so two steps can lie closer than the digits a log prints. Only when no
    step holds it is a value read to those digits (:func:`same_step_value`),
    and then only if the steps it reaches hold one value: a value the printed
    digits join to two values the run holds apart names neither.
    """

    def having(same: Any) -> list[int]:
        return [
            index
            for index, values in enumerate(step_values)
            if all(
                (key := step_axis_key(axis, values)) in values and same(values[key], value)
                for axis, value in wanted.items()
            )
        ]

    held = having(_same_held_value)
    if held:
        return held
    near = having(same_step_value)
    for axis in wanted:
        reached = [step_values[i][step_axis_key(axis, step_values[i])] for i in near]
        if any(not _same_held_value(reached[0], other) for other in reached[1:]):
            return []
    return near


def step_with(
    step_values: Sequence[Mapping[str, Any]], parameters: Mapping[str, Any]
) -> int | None:
    """The index of the first step whose values are ``parameters``, no more
    and no fewer (:func:`steps_with`); None when no step's are."""
    return next(
        (
            index
            for index in steps_with(step_values, parameters)
            if len(step_values[index]) == len(parameters)
        ),
        None,
    )


def step_value_text(value: Any) -> str:
    """A step value as a refusal lists it: a number to ten significant digits,
    which names the same step when passed back, anything else as it is."""
    number = step_number(value)
    return str(value) if number is None else format_spice_value(number)


def step_values_text(values: Mapping[str, Any]) -> str:
    """One step's values as a ``.step`` line writes them (``r=1000 c=1e-07``);
    ``none`` for a step of no values."""
    return " ".join(f"{axis}={step_value_text(value)}" for axis, value in values.items()) or "none"


#: How many of a run's steps, values or axes a refusal lists.
_LISTED = 20


def listed(items: Sequence[str]) -> str:
    """``items`` as a refusal lists them: the first 20, and how many in all
    when there are more."""
    shown = ", ".join(items[:_LISTED])
    if len(items) > _LISTED:
        shown += f", ... ({len(items)} in all)"
    return shown


def listed_steps(step_values: Sequence[Mapping[str, Any]]) -> str:
    """A run's steps as a refusal lists them, each as :func:`step_values_text`
    writes it."""
    return listed([step_values_text(values) for values in step_values])
