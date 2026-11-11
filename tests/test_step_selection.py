"""Which ``.step`` iteration a step selector reads, and what each step is called.

A step value is read the way every other value of a request is: ``'2k'`` is
2000 whichever way it is written. It selects only a step with that value, and
a value no step has is refused with the ones the run has, on every route that
selects a step by its value. The step axes come from the recorded runs in
``tests/fixtures/ltspice_recorded``, whose names are the deck's ``.step``
names whether the run kept its step values in its log or in its raw.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.format import SAME_VALUE_RTOL
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.metrics import AT_SNAP_RTOL, snap_match
from ltspice_mcp.lib.recipes import StepSelector
from ltspice_mcp.lib.step_values import listed_steps, same_step_value, step_with
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests._ltspice_recorded import header_fields
from tests.conftest import SyncApi, stage_recorded_fixture

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


def test_a_number_selects_a_step_it_equals_to_the_shared_tolerance():
    """A value read back from a run is the number its deck wrote to about
    seven digits (a raw stores most samples in four bytes): within
    ``SAME_VALUE_RTOL`` it is the same value, and outside it another one."""
    stored = float(np.float32(1e-7))
    assert stored != 1e-7
    steps = [{"c": stored}, {"c": 2e-7}]
    assert StepSelector(axis="c", value="100n").pick(steps) == 0
    assert StepSelector(axis="c", value=1e-7 * (1 + SAME_VALUE_RTOL / 2)).pick(steps) == 0
    with pytest.raises(ResultError, match=r"its values are 1.000000012e-07, 2e-07;"):
        StepSelector(axis="c", value=1e-7 * (1 + 2 * SAME_VALUE_RTOL)).pick(steps)


def test_an_at_read_is_exact_over_a_wider_gap_than_a_step_value_selects():
    """The two ask ``same_value`` with different tolerances: ``exact_match``
    flags the ``value`` recipe's nearest-sample read only when it moved the
    point far enough to matter (``metrics.AT_SNAP_RTOL``), whereas a step
    value names one of the run's own values."""
    assert AT_SNAP_RTOL > SAME_VALUE_RTOL
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


def _step_or_refusal(reply: dict[str, Any]) -> int | str:
    """The one step an ``analyze_results`` reply read, or why it refused."""
    if reply.get("failures"):
        (failure,) = reply["failures"]
        return failure["message"]
    (row,) = reply["results"]["k"]["per_run"]["items"]
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


async def _every_tool_route(
    state: SessionState, raw: Path, steps: list[dict[str, Any]], value: Any
) -> dict[str, int | str]:
    """The step each route reads of ``raw`` for ``{axis:'r', value}``, or the
    message it refuses it with. The measurements recipe reads the steps of
    the log beside the raw, the others those of the raw."""
    try:
        selected: int | str = StepSelector(axis="r", value=value).pick(steps)
    except ResultError as refused:
        selected = str(refused)
    return {
        "selector": selected,
        "analyze_results raw recipe": await _analyzed_step(
            state, raw, {"metric": "signal_stats", "signal": "V(out)"}, value
        ),
        "analyze_results measurements": await _analyzed_step(
            state, raw, {"metric": "measurements"}, value
        ),
        "plot_waveform": await _plotted_step(state, raw, value),
    }


def _step_tran(build: str, directory: Path) -> tuple[Path, list[dict[str, Any]]]:
    raw = rec.stage_run(build, "raw/step_tran", directory / build)
    steps = [
        dict(step.parameters)
        for step in rec.decode(build, "raw/step_tran", directory / "decoded").raw.descriptor.steps
    ]
    assert steps == [{"r": 1000.0}, {"r": 2000.0}, {"r": 4000.0}]
    return raw, steps


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_step_value_no_step_has_is_refused_on_every_route(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``raw/step_tran`` steps r over 1k, 2k and 4k. '6.8k' is none of them:
    every route refuses it and lists the three, where each used to read the
    run at 4k, the nearest, and say nothing."""
    raw, steps = _step_tran(build, work_dir)
    routes = await _every_tool_route(state_no_sim, raw, steps, "6.8k")
    assert len(routes) == 4
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
    build: str, value: Any, index: int, state_no_sim: SessionState, work_dir: Path
):
    raw, steps = _step_tran(build, work_dir)
    routes = await _every_tool_route(state_no_sim, raw, steps, value)
    assert routes == dict.fromkeys(routes, index)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_python_api_reads_a_step_value_as_the_tools_do(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``api.analyze_results`` takes the same selector. ``RawResult.compare``
    pairs a step with the other run's by ``step_with``, and refuses with the
    steps the other run has when none matches."""
    api = SyncApi(state_no_sim)
    raw, _ = _step_tran(build, work_dir)

    def analyzed(value: Any) -> int | str:
        return _step_or_refusal(
            api.analyze_results(
                sources=[{"path": str(raw), "label": "dut"}],
                recipes=[{"key": "k", "metric": "signal_stats", "signal": "V(out)"}],
                step={"axis": "r", "value": value},
                include={"per_run": True},
            )
        )

    assert _refused_6800(analyzed("6.8k")) == "refused"
    assert [analyzed(value) for value in ("1k", 1e3, "2k", 4e3)] == [0, 0, 1, 2]

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
