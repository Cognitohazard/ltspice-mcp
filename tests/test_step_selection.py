"""Which ``.step`` iteration a step selector reads, and what each step is called.

A step value is read the way every other value of a request is: ``'2k'`` is
2000 whichever way it is written. It selects only a step with that value, to
the six significant digits a log prints a step value with, and a value no
step has is refused with the ones the run has, on every route that selects a
step by its value. The step axes come from the recorded runs in
``tests/fixtures/ltspice_recorded``, whose names are the deck's ``.step``
names whether the run kept its step values in its log or in its raw.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.format import parse_spice_value, same_value
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.metrics import AT_SNAP_RTOL, snap_match
from ltspice_mcp.lib.recipes import StepSelector
from ltspice_mcp.lib.step_values import (
    STEP_VALUE_DIGITS,
    listed_steps,
    printed_half_unit,
    same_step_value,
    step_with,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests._ltspice_recorded import header_fields
from tests.conftest import (
    LIVENESS_S,
    SyncApi,
    dispatched,
    replaying_simulator,
    stage_recorded_fixture,
)

_SUFFIXES = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "Meg"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)


def engineering(value: float) -> str:
    """``value`` as a deck writes it: a mantissa from 1 to 1000 and a suffix."""
    if value == 0:
        return "0"
    scale, suffix = next(
        ((scale, suffix) for scale, suffix in _SUFFIXES if abs(value) >= scale), _SUFFIXES[-1]
    )
    return f"{value / scale:.12g}{suffix}"


def in_kilo(value: float) -> str:
    """``value`` in thousands, so every value is written with a suffix."""
    return f"{value / 1e3:.12g}k"


def stepped_runs() -> list[tuple[str, str]]:
    """Every recorded run that ended by itself and whose raw says it is stepped."""
    runs = []
    for build in rec.BUILDS:
        for case_id, entry in sorted(rec.manifest(build)["cases"].items()):
            output = entry["outputs"].get(f"{case_id}.raw")
            if output is None or output["bytes"] == 0 or entry["stopped"]:
                continue
            if "stepped" in header_fields(build, case_id)["Flags"][0].split():
                runs.append((build, case_id))
    return runs


STEPPED = stepped_runs()


def test_the_corpus_has_stepped_runs_on_every_build():
    assert {build for build, _ in STEPPED} == set(rec.BUILDS)
    assert len(STEPPED) >= 20


@pytest.mark.parametrize(("build", "case_id"), STEPPED)
def test_a_step_value_in_spice_notation_selects_the_step_its_number_does(
    build: str, case_id: str, tmp_path: Path
):
    steps = [
        dict(step.parameters) for step in rec.decode(build, case_id, tmp_path).raw.descriptor.steps
    ]
    numeric = [
        (axis, value)
        for values in steps
        for axis, value in values.items()
        if isinstance(value, float) and math.isfinite(value)
    ]
    if not numeric:
        pytest.skip(f"{build} {case_id}: the run names no step values")
    for axis, value in numeric:
        index = StepSelector(axis=axis, value=value).pick(steps)
        assert steps[index][axis] == value
        for written in (engineering(value), in_kilo(value), repr(value)):
            assert StepSelector(axis=axis, value=written).pick(steps) == index, (axis, written)


@pytest.mark.parametrize(("build", "case_id"), STEPPED)
def test_a_text_value_no_step_has_is_refused_with_the_values_it_has(
    build: str, case_id: str, tmp_path: Path
):
    steps = [
        dict(step.parameters) for step in rec.decode(build, case_id, tmp_path).raw.descriptor.steps
    ]
    if not steps or not steps[0]:
        pytest.skip(f"{build} {case_id}: the run names no step values")
    axis = next(iter(steps[0]))
    with pytest.raises(ResultError, match=rf"no step has {axis} = 'fast': its values are"):
        StepSelector(axis=axis, value="fast").pick(steps)


def test_an_axis_the_run_does_not_step_is_refused_with_the_axes_it_does():
    steps = [{"r": 1.0, "c": 1e-9}, {"r": 2.0, "c": 1e-9}]
    with pytest.raises(ResultError, match=r"step axis 'v' is not present.*the run steps c, r"):
        StepSelector(axis="v", value=1).pick(steps)


def test_a_text_value_selects_a_step_whose_value_is_that_text():
    steps = [{"model": "fast"}, {"model": "slow"}]
    assert StepSelector(axis="model", value="slow").pick(steps) == 1


def test_a_number_selects_only_a_step_it_equals():
    """It used to select the nearest step, so '25' read the run at 22 and
    '1Meg' at 680 with nothing said."""
    steps = [{"r": 1.0}, {"r": 22.0}, {"r": 680.0}]
    assert StepSelector(axis="r", value="0.68k").pick(steps) == 2
    assert StepSelector(axis="r", value=22).pick(steps) == 1
    with pytest.raises(ResultError, match=r"no step has r = '25' \(read as 25\): its values are"):
        StepSelector(axis="r", value="25").pick(steps)
    with pytest.raises(ResultError, match=r"no step has r = '1Meg' \(read as 1000000\)"):
        StepSelector(axis="r", value="1Meg").pick(steps)
    with pytest.raises(ResultError, match=r"no step has r = 25\.0: its values are 1, 22, 680;"):
        StepSelector(axis="r", value=25.0).pick(steps)


def test_a_number_selects_a_step_to_the_digits_its_value_was_printed_with():
    """A stepped sweep's step values are its log's ``.step`` lines, which
    print six significant digits, so ``.step param r list 1k
    2.1544346900318838k`` would log ``r=2154.43``. The deck's own value, and
    that value to seven digits, select the step; a value the log would print
    as another number does not. The deck's value used to be refused: it is
    two parts in a million from the printed one."""
    steps = parse_step_iterations(text=".step r=1000\n.step r=2154.43\n.step r=4641.59\n")
    assert steps == [{"r": 1000.0}, {"r": 2154.43}, {"r": 4641.59}]
    assert StepSelector(axis="r", value=1000 * 10 ** (1 / 3)).pick(steps) == 1
    assert StepSelector(axis="r", value="2.154435k").pick(steps) == 1
    assert StepSelector(axis="r", value=1000 * 10 ** (2 / 3)).pick(steps) == 2
    with pytest.raises(
        ResultError,
        match=(
            r"no step has r = '2154\.436' \(read as 2154\.436\): "
            r"its values are 1000, 2154\.43, 4641\.59; name one of them"
        ),
    ):
        StepSelector(axis="r", value="2154.436").pick(steps)


def test_a_step_value_a_raw_stores_in_four_bytes_is_selected_by_the_decks():
    """Where a raw stores one point a step it holds each step's values as its
    first variables, the first in eight bytes and any other in four, which
    keep about seven digits."""
    stored = float(np.float32(1e-7))
    assert stored != 1e-7
    steps = [{"c": stored}, {"c": 2e-7}]
    assert StepSelector(axis="c", value="100n").pick(steps) == 0
    with pytest.raises(ResultError, match=r"its values are 1.000000012e-07, 2e-07;"):
        StepSelector(axis="c", value="100.001n").pick(steps)


def test_a_step_value_is_matched_to_the_digits_printed_not_to_the_shared_tolerance():
    """``format.same_value`` holds two values to a part in a million
    (``SAME_VALUE_RTOL``), which a value a raw stores keeps. A log prints a
    step value to six significant digits, which move it by up to five parts in
    a million, so a step value is matched to those digits instead. No
    tolerance that is a fixed share of the value can do this: one wide enough
    to take the deck's value for ``2154.43`` also takes ``100.001`` for
    ``100.002``, which the log prints apart."""
    exact = 1000 * 10 ** (1 / 3)
    assert not same_value(exact, 2154.43)
    assert same_step_value(exact, 2154.43)
    assert same_value(100.001, 100.002, rtol=1e-5)
    assert not same_step_value(100.001, 100.002)
    # Ten is printed as 10 from 9.999995 up, and 9.99999 is printed apart.
    assert same_step_value(9.999996, 10.0)
    assert not same_step_value(9.99999, 10.0)
    assert not same_step_value(0.0, 1e-30)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_each_value_a_deck_steps_selects_its_step_from_what_its_log_printed(
    build: str, tmp_path: Path
):
    """``raw/step_digits`` steps r over values the deck writes to seventeen
    digits and two that differ only in the sixth. The log prints each to at
    least the six digits a step value is matched to, and the deck's own value
    selects its step."""
    if not rec.has(build, "raw/step_digits.log"):
        pytest.skip(rec.CASES.behaviours["step-value-digits"].pending)
    deck = (rec.INPUTS / "raw/step_digits.cir").read_text(encoding="utf-8")
    (line,) = [card for card in deck.splitlines() if card.startswith(".step ")]
    written = [parse_spice_value(token) for token in line.partition(" list ")[2].split()]
    steps = [
        dict(step.parameters)
        for step in rec.decode(build, "raw/step_digits", tmp_path).raw.descriptor.steps
    ]
    assert len(steps) == len(written)
    for index, value in enumerate(written):
        printed = steps[index]["r"]
        assert isinstance(printed, float), steps[index]
        assert abs(printed - value) <= printed_half_unit(value), (printed, value)
        assert StepSelector(axis="r", value=value).pick(steps) == index


def test_an_at_read_is_exact_over_a_wider_gap_than_a_step_value_selects():
    """``exact_match`` flags the ``value`` recipe's nearest-sample read only
    when it moved the point far enough to matter (``metrics.AT_SNAP_RTOL``),
    whereas a step value names one of the run's own values, to the digits it
    was printed with. Those move a value by at most half a unit in the sixth
    digit of ``1.00000``."""
    widest = printed_half_unit(1.0)
    assert widest == pytest.approx(0.5 * 10.0 ** (1 - STEP_VALUE_DIGITS))
    assert widest < AT_SNAP_RTOL
    assert snap_match(1.0, 1.0005)
    assert not snap_match(1.0, 1.002)
    with pytest.raises(ResultError, match=r"no step has r = 1\.0: its values are 1\.0005;"):
        StepSelector(axis="r", value=1.0).pick([{"r": 1.0005}])


@pytest.mark.parametrize(("build", "case_id"), STEPPED)
def test_every_value_a_refusal_lists_selects_a_step_of_that_value(
    build: str, case_id: str, tmp_path: Path
):
    steps = [
        dict(step.parameters) for step in rec.decode(build, case_id, tmp_path).raw.descriptor.steps
    ]
    if not steps or not steps[0]:
        pytest.skip(f"{build} {case_id}: the run names no step values")
    for axis in steps[0]:
        with pytest.raises(ResultError) as refused:
            StepSelector(axis=axis, value="fast").pick(steps)
        listed = str(refused.value).split("its values are ")[1].removesuffix("; name one of them")
        written = listed.split(", ")
        assert len(written) == len({str(values[axis]) for values in steps}), listed
        for text in written:
            index = StepSelector(axis=axis, value=text).pick(steps)
            assert same_step_value(steps[index][axis], text), (axis, text)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_raws_step_filter_matches_a_value_as_a_selector_does(build: str, tmp_path: Path):
    """``get_steps(**values)``, the filter a decoded raw keeps for spicelib's
    readers, asks the same rule: it compared values with ``==``, so ``'2k'``
    matched no step of 2000."""
    raw = rec.decode(build, "raw/step_tran", tmp_path).raw
    assert raw.get_steps() == [0, 1, 2]
    assert raw.get_steps(r=2000.0) == [1]
    assert raw.get_steps(r="2k") == [1]
    assert raw.get_steps(R=2e3) == [1]
    assert raw.get_steps(r="6.8k") == []


def test_a_step_is_paired_by_values_each_the_same_as_a_selector_reads_it():
    """``RawResult.compare`` pairs a step with the other run's step of the
    same parameters by ``step_with``: names found as a selector finds them,
    values the same to the shared tolerance."""
    other = [{"rd": 1e3, "c": float(np.float32(1e-7))}, {"rd": 2e3, "c": float(np.float32(1e-7))}]
    assert step_with(other, {"RD": 2000.0, "c": 1e-7}) == 1
    assert step_with(other, {"rd": 2000.0}) is None
    assert step_with(other, {"rd": 4000.0, "c": 1e-7}) is None
    assert listed_steps(other) == "rd=1000 c=1.000000012e-07, rd=2000 c=1.000000012e-07"
    # A step whose log printed its value, paired with one that holds it whole.
    printed = parse_step_iterations(text=".step r=1000\n.step r=2154.43\n")
    assert step_with(printed, {"r": 1000 * 10 ** (1 / 3)}) == 1


@pytest.mark.asyncio
async def test_analyze_results_reads_the_step_a_suffixed_value_names(
    state_no_sim: SessionState, work_dir: Path
):
    """``.step r list 1 22 680``: '0.68k' is the third step, as 680 is."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    rows = []
    for value in (680, "0.68k"):
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "dut"}],
                "recipes": [{"key": "s", "metric": "signal_stats", "signal": "V(out)"}],
                "step": {"axis": "r", "value": value},
                "include": {"fields": ["step_index", "step_values", "value.rms"]},
            }
        )
        reply = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert reply is not None
        (row,) = reply["results"]["s"]["values"]
        rows.append(row)
    assert rows[0] == rows[1]
    assert (rows[1]["step_index"], rows[1]["step_values"]) == (2, {"r": 680.0})


# --------------------------------------------------------------------------
# A stepped temperature has one name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("case_id", ["raw/step_temp", "raw/step_temp_op"])
def test_a_stepped_temperature_is_named_as_the_deck_steps_it(
    build: str, case_id: str, tmp_path: Path
):
    """The deck says ``.step temp list -40 27 125``. The sweep's steps come
    from its log's ``.step temp=-40°`` lines, the operating point's from its
    raw's ``temperature`` variable; both are ``temp``."""
    deck = (rec.INPUTS / f"{case_id}.cir").read_text(encoding="utf-8")
    assert ".step temp list -40 27 125" in deck
    raw = rec.decode(build, case_id, tmp_path).raw
    assert raw.descriptor.step_status == "matched"
    assert [dict(step.parameters) for step in raw.descriptor.steps] == [
        {"temp": -40.0},
        {"temp": 27.0},
        {"temp": 125.0},
    ]


async def _reply(state: SessionState, raw: Path, **selection: Any) -> dict[str, Any]:
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "i", "metric": "value", "expr": "I(R1)", "at": "0"}]
            if raw.stem == "step_temp"
            else [{"key": "i", "metric": "value", "expr": "I(R1)"}],
            "include": {"fields": ["step_index", "step_values", "value.value"]},
            **selection,
        }
    )
    reply = (await handle_analyze_results(request, state)).structured_content
    assert reply is not None
    assert reply.get("failures", []) == [], reply.get("failures", [])
    return reply["results"]["i"]


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("case_id", ["raw/step_temp", "raw/step_temp_op"])
@pytest.mark.parametrize("axis", ["temp", "temperature"])
async def test_either_name_of_a_stepped_temperature_selects_and_groups_its_steps(
    build: str, case_id: str, axis: str, state_no_sim: SessionState, work_dir: Path
):
    raw = rec.stage_run(build, case_id, work_dir / build)
    (row,) = (await _reply(state_no_sim, raw, step={"axis": axis, "value": 27}))["values"]
    assert (row["step_index"], row["step_values"]) == (1, {"temp": 27.0})
    groups = (await _reply(state_no_sim, raw, all_steps=True, group_by=[axis]))["groups"]
    assert [(group["by"], group["count"]) for group in groups] == [
        ({axis: -40.0}, 1),
        ({axis: 27.0}, 1),
        ({axis: 125.0}, 1),
    ]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_an_axis_named_as_the_deck_spells_it_selects_the_step(build: str):
    """The deck steps ``Rv`` (``.step param Rv list 1k 2k``) and both builds
    print the name lower-cased in the log, ``.step rv=2000``
    (``log/meas_when_threshold_tran_step``). SPICE compares names without
    case, so the deck's spelling names the axis."""
    steps = parse_step_iterations(rec.recorded(build, "log/meas_when_threshold_tran_step.log"))
    assert steps == [{"rv": 1000.0}, {"rv": 2000.0}]
    assert StepSelector(axis="Rv", value="2k").pick(steps) == 1
    assert StepSelector(axis="RV", value=1000).pick(steps) == 0


# --------------------------------------------------------------------------
# An axis named in another case reads the deck's axis on every route
# --------------------------------------------------------------------------


async def _signal_stats(state: SessionState, raw: Path, **selection: Any) -> dict[str, Any]:
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "s", "metric": "signal_stats", "signal": "V(out)"}],
            "include": {"fields": ["step_index", "step_values", "value.max"]},
            **selection,
        }
    )
    reply = (await handle_analyze_results(request, state)).structured_content
    assert reply is not None
    assert reply.get("failures", []) == [], reply.get("failures", [])
    return reply["results"]["s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_an_axis_named_in_another_case_reads_the_same_step_on_every_route(
    build: str, state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """``raw/step_tran`` steps ``r`` (``.step param r list 1k 2k 4k``) and the
    caller names it ``R``. The selector, analyze_results' ``step`` and
    ``group_by``, and plot_waveform's ``step`` each read it as the deck's
    ``r``, and the step they pick peaks at the same voltage."""
    deck = (rec.INPUTS / "raw/step_tran.cir").read_text(encoding="utf-8")
    assert ".step param r list 1k 2k 4k" in deck
    steps = [
        dict(step.parameters)
        for step in rec.decode(build, "raw/step_tran", tmp_path).raw.descriptor.steps
    ]
    assert steps == [{"r": 1000.0}, {"r": 2000.0}, {"r": 4000.0}]
    assert StepSelector(axis="R", value="2k").pick(steps) == 1

    raw = rec.stage_run(build, "raw/step_tran", work_dir / build)
    selected = await _signal_stats(state_no_sim, raw, step={"axis": "R", "value": "2k"})
    (row,) = selected["values"]
    assert (row["step_index"], row["step_values"]) == (1, {"r": 2000.0})
    grouped = await _signal_stats(state_no_sim, raw, all_steps=True, group_by=["R"])
    assert [(group["by"], group["count"]) for group in grouped["groups"]] == [
        ({"R": 1000.0}, 1),
        ({"R": 2000.0}, 1),
        ({"R": 4000.0}, 1),
    ]

    plotted = await handle_plot_waveform(
        PlotWaveformInput.model_validate(
            {
                "path": str(raw),
                "signals": ["V(out)"],
                "step": {"axis": "R", "value": "2k"},
                "open": False,
            }
        ),
        state_no_sim,
    )
    assert plotted.structured_content is not None
    (trace,) = plotted.structured_content["traces"]
    assert trace["step"] == 1
    assert trace["max"] == pytest.approx(row["value"]["max"], rel=1e-12)


# --------------------------------------------------------------------------
# A value no step has is refused on every route, and a step's value written
# another way is read on every route
# --------------------------------------------------------------------------

#: What every route says of ``raw/step_tran``'s steps when refusing a value.
STEP_TRAN_VALUES = "its values are 1000, 2000, 4000; name one of them"


def _step_or_refusal(reply: dict[str, Any], key: str = "k") -> int | str:
    """The one step an ``analyze_results`` reply read for recipe ``key``, or
    why it refused. A refusal names the run's steps, which every recipe
    over the run shares, so the reply holds one message however many
    recipes it refused."""
    if reply.get("failures"):
        (message,) = {failure["message"] for failure in reply["failures"]}
        return message
    (row,) = reply["results"][key]["per_run"]["items"]
    return row["step_index"]


async def _analyzed_step(
    state: SessionState, raw: Path, recipe: dict[str, Any], value: Any
) -> int | str:
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "k", **recipe}],
            "step": {"axis": "r", "value": value},
            "include": {"per_run": True},
        }
    )
    reply = (await handle_analyze_results(request, state)).structured_content
    assert reply is not None
    return _step_or_refusal(reply)


async def _plotted_step(state: SessionState, raw: Path, value: Any) -> int | str:
    try:
        plotted = await handle_plot_waveform(
            PlotWaveformInput.model_validate(
                {
                    "path": str(raw),
                    "signals": ["V(out)"],
                    "step": {"axis": "r", "value": value},
                    "open": False,
                }
            ),
            state,
        )
    except ResultError as refused:
        return str(refused)
    assert plotted.structured_content is not None
    (trace,) = plotted.structured_content["traces"]
    return trace["step"]


_SIGNAL_STATS = {"metric": "signal_stats", "signal": "V(out)"}
_MEASUREMENTS = {"metric": "measurements"}


async def _attached_steps(state: SessionState, deck: Path, value: Any) -> dict[str, int | str]:
    """The step ``run_experiments``' attached analysis reads of a job of
    ``deck`` for ``{axis:'r', value}``, or the message it refuses it with,
    as the job's receipt shows it and as ``jobs(status)`` returns it."""
    receipt = await dispatched(
        state,
        "run_experiments",
        circuits=[{"path": str(deck), "id": "rc"}],
        analyze={
            "recipes": [{"key": "k", **_SIGNAL_STATS}, {"key": "m", **_MEASUREMENTS}],
            "step": {"axis": "r", "value": value},
            "include": {"per_run": True},
        },
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    status = await dispatched(state, "jobs", action="status", job_id=receipt["job_id"])
    routes: dict[str, int | str] = {}
    for reply, shown in ((receipt, "run_experiments receipt"), (status, "jobs status")):
        assert reply["analysis"]["status"] == "completed", reply["analysis"]
        for key, recipe in (("k", "raw recipe"), ("m", "measurements")):
            routes[f"{shown}, {recipe}"] = _step_or_refusal(reply["analysis"]["result"], key)
    return routes


@dataclass(frozen=True)
class _StagedRun:
    """A recorded run staged for every route to read: its raw with its log
    beside it, the deck it ran, and its steps as the decoder reads them."""

    raw: Path
    deck: Path
    steps: list[dict[str, Any]]


async def _every_tool_route(
    state: SessionState, run: _StagedRun, value: Any
) -> dict[str, int | str]:
    """The step each route reads of ``run`` for ``{axis:'r', value}``, or the
    message it refuses it with. The measurements recipe reads the steps of
    the log beside the raw, the others those of the raw. A job of the run's
    deck hands back the run (``replaying_simulator``), and its attached
    analysis asks both recipes."""
    try:
        selected: int | str = StepSelector(axis="r", value=value).pick(run.steps)
    except ResultError as refused:
        selected = str(refused)
    return {
        "selector": selected,
        "analyze_results raw recipe": await _analyzed_step(state, run.raw, _SIGNAL_STATS, value),
        "analyze_results measurements": await _analyzed_step(state, run.raw, _MEASUREMENTS, value),
        "plot_waveform": await _plotted_step(state, run.raw, value),
        **await _attached_steps(state, run.deck, value),
    }


def _stage_step_tran(
    build: str, directory: Path, values: tuple[str, str] = ("2k", "4k")
) -> _StagedRun:
    """``raw/step_tran`` staged in ``directory``: ``.step param r list 1k 2k
    4k``. ``values`` puts others in place of 2k and 4k, in the deck and in
    each ``.step`` line of the log as the log prints them (C's ``%g``); the
    raw does not hold them."""
    raw = rec.stage_run(build, "raw/step_tran", directory / build)
    log, deck = raw.with_suffix(".log"), raw.with_suffix(".cir")
    shutil.copyfile(rec.INPUTS / "raw/step_tran.cir", deck)
    for path, replace in (
        (deck, {"list 1k 2k 4k": f"list 1k {values[0]} {values[1]}"}),
        (
            log,
            {
                f".step r={old:g}": f".step r={parse_spice_value(new):g}"
                for old, new in zip((2000.0, 4000.0), values, strict=True)
            },
        ),
    ):
        text = path.read_bytes()
        for old, new in replace.items():
            assert text.count(old.encode()) == 1, (path, old)
            text = text.replace(old.encode(), new.encode())
        path.write_bytes(text)
    steps = [
        dict(step.parameters)
        for step in rec.decode_file(raw, directory / "decoded", log=log).raw.descriptor.steps
    ]
    return _StagedRun(raw, deck, steps)


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_step_value_no_step_has_is_refused_on_every_route(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``raw/step_tran`` steps r over 1k, 2k and 4k. '6.8k' is none of them:
    every route refuses it and lists the three, where each used to read the
    run at 4k, the nearest, and say nothing."""
    run = _stage_step_tran(build, work_dir)
    assert run.steps == [{"r": 1000.0}, {"r": 2000.0}, {"r": 4000.0}]
    replaying_simulator(monkeypatch, run.raw, run.raw.with_suffix(".log"))
    routes = await _every_tool_route(state_with_sim, run, "6.8k")
    assert len(routes) == 8
    read = {route: _refused_6800(found) for route, found in routes.items()}
    assert read == dict.fromkeys(routes, "refused"), routes


def _refused_6800(found: int | str) -> int | str:
    """``refused`` when a route refused '6.8k' with ``raw/step_tran``'s
    values, else what it read or said."""
    if isinstance(found, str) and "no step has r = '6.8k' (read as 6800)" in found:
        return "refused" if STEP_TRAN_VALUES in found else found
    return found


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    ("value", "index"), [("1k", 0), (1e3, 0), (1000, 0), ("0.002Meg", 1), ("4000.0", 2)]
)
async def test_a_step_value_written_another_way_is_read_on_every_route(
    build: str,
    value: Any,
    index: int,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run = _stage_step_tran(build, work_dir)
    replaying_simulator(monkeypatch, run.raw, run.raw.with_suffix(".log"))
    routes = await _every_tool_route(state_with_sim, run, value)
    assert len(routes) == 8
    assert routes == dict.fromkeys(routes, index)


#: Two values ``.step param r list 1k 2.1544346900318838k
#: 4.6415888336127789k`` steps, which a log prints to six significant digits
#: as C's ``%g`` writes a number (``raw/step_digits`` records what each build
#: prints for such a value).
_UNROUND = ("2.1544346900318838k", "4.6415888336127789k")
_UNROUND_VALUES = "its values are 1000, 2154.43, 4641.59; name one of them"


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    ("value", "read"),
    [
        (1000 * 10 ** (1 / 3), 1),
        ("2.1544346900318838k", 1),
        ("2154.435", 1),
        (1000 * 10 ** (2 / 3), 2),
        ("2154.436", f"no step has r = '2154.436' (read as 2154.436): {_UNROUND_VALUES}"),
        (4641.6, f"no step has r = 4641.6: {_UNROUND_VALUES}"),
    ],
)
async def test_a_step_value_is_read_to_the_digits_its_log_printed_on_every_route(
    build: str,
    value: Any,
    read: int | str,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A stepped sweep's step values are its log's ``.step`` lines, which
    print six significant digits: ``r=2154.43`` for the deck's
    2154.4346900318838. The deck's value selects its step on every route, as
    does that value to seven digits; a value the log would print as another
    number is refused. Every route refused the deck's value, which is two
    parts in a million from the printed one."""
    run = _stage_step_tran(build, work_dir, _UNROUND)
    assert run.steps == [{"r": 1000.0}, {"r": 2154.43}, {"r": 4641.59}]
    replaying_simulator(monkeypatch, run.raw, run.raw.with_suffix(".log"))
    routes = await _every_tool_route(state_with_sim, run, value)
    assert len(routes) == 8
    assert routes == dict.fromkeys(routes, read)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_python_api_reads_a_step_value_as_the_tools_do(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``api.analyze_results`` takes the same selector, and a decoded raw's
    ``get_steps`` filter asks the same rule. ``RawResult.compare`` pairs a
    step with the other run's by ``step_with``, and refuses with the steps
    the other run has when none matches."""
    api = SyncApi(state_no_sim)
    raw = _stage_step_tran(build, work_dir).raw
    printed = _stage_step_tran(build, work_dir / "printed", _UNROUND)

    def analyzed(value: Any, run: Path = raw) -> int | str:
        return _step_or_refusal(
            api.analyze_results(
                sources=[{"path": str(run), "label": "dut"}],
                recipes=[{"key": "k", "metric": "signal_stats", "signal": "V(out)"}],
                step={"axis": "r", "value": value},
                include={"per_run": True},
            )
        )

    assert _refused_6800(analyzed("6.8k")) == "refused"
    assert [analyzed(value) for value in ("1k", 1e3, "2k", 4e3)] == [0, 0, 1, 2]
    assert analyzed(1000 * 10 ** (1 / 3), printed.raw) == 1
    assert analyzed("2154.436", printed.raw) == (
        f"no step has r = '2154.436' (read as 2154.436): {_UNROUND_VALUES}"
    )
    decoded = rec.decode_file(
        printed.raw, work_dir / "printed-decoded", log=printed.raw.with_suffix(".log")
    ).raw
    assert decoded.get_steps(r=1000 * 10 ** (1 / 3)) == [1]
    assert decoded.get_steps(r="2154.436") == []

    stepped = api.load_raw(path=raw)
    other_build = next(other for other in rec.BUILDS if other != build)
    across = api.load_raw(path=rec.stage_run(other_build, "raw/step_tran", work_dir / other_build))
    assert stepped.compare(across, step=2) == stepped.compare(across, step=2, other_step=2)
    dc = api.load_raw(path=rec.stage_run(build, "raw/step_dc", work_dir / build))
    assert dc.steps == [{"r": 1000.0}, {"r": 3000.0}]
    with pytest.raises(
        ResultError,
        match=(
            r"No step of the other result has this step's parameters \(r=2000\): "
            r"its steps are r=1000, r=3000; pass other_step\."
        ),
    ):
        stepped.compare(dc, step=1)
