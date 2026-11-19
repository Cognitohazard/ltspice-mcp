"""Recipes for loop-gain work: one reader of a signal expression (``-``, ``A/B``,
``1/A``) behind every recipe that reads a ratio, ``tone`` of a ratio and at
each run's own frequency, and ``stability`` on a frequency response analysis
checked against the margin LTspice printed for it.

Each answer is read through ``analyze_results`` and through the Python API,
which share the evaluator but not the reply's rendering.
"""

from __future__ import annotations

import ast
import asyncio
import cmath
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import metrics
from ltspice_mcp.lib.recipes import RECIPE_MODELS
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests import _source_scan
from tests.conftest import SyncApi, make_experiment_job, stage_recorded_fixture
from tests.test_analyze_results import _two_tone_raw

Reader = Callable[[SessionState, dict[str, Any]], Any]


async def _through_tool(state: SessionState, request: dict[str, Any]) -> dict[str, Any]:
    reply = await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state)
    assert reply.structured_content is not None
    return reply.structured_content


async def _through_api(state: SessionState, request: dict[str, Any]) -> dict[str, Any]:
    # SyncApi runs each call on a loop of its own, which cannot nest in this one.
    return await asyncio.to_thread(SyncApi(state).analyze_results, **request)


ROUTES = pytest.mark.parametrize("read", [_through_tool, _through_api], ids=["tool", "api"])


def _rows(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    assert "failures" not in data, data.get("failures")
    return data["results"][key]["values"]


def _write_tran(path: Path, t: np.ndarray, traces: dict[str, np.ndarray]) -> None:
    from spicelib.raw.raw_write import RawWrite, Trace

    writer = RawWrite(plot_name="Transient Analysis")
    writer.add_trace(Trace("time", t, whattype="time"))
    for name, values in traces.items():
        writer.add_trace(Trace(name, values, whattype="voltage"))
    writer.save(path)


def _phasor(value: dict[str, Any]) -> complex:
    """A tone row as the complex amplitude of its sinusoid, referred to sin."""
    return cmath.rect(value["amplitude"], math.radians(value["phase_deg"]))


# ---------------------------------------------------------------------------
# tone at each run's own frequency
# ---------------------------------------------------------------------------

#: Three cases of one job, each injecting its own frequency: (FINJ as the case
#: assigns it, amplitude, phase in degrees).
_CASES = [("1k", 1.0, 30.0), (2500.0, 0.5, -45.0), ("4k", 0.25, 120.0)]


def _injection_job(state: SessionState, work_dir: Path) -> str:
    job = make_experiment_job(state, job_id="exp_finj", count=len(_CASES))
    t = np.linspace(0.0, 4e-3, 40_001)
    for case, (finj, amplitude, phase) in zip(job.cases, _CASES, strict=True):
        frequency = 1e3 * float(str(finj).rstrip("k")) if isinstance(finj, str) else finj
        case.assignments = {"FINJ": finj}
        out = 0.1 + amplitude * np.sin(2 * np.pi * frequency * t + math.radians(phase))
        assert case.raw_file is not None
        _write_tran(case.raw_file, t, {"V(out)": out})
    return job.job_id


@ROUTES
async def test_one_tone_recipe_reads_each_case_at_its_own_frequency(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """A sweep's cases each inject their own frequency, named by an assignment;
    one recipe reads every case at the frequency that case assigns."""
    job_id = _injection_job(state_no_sim, work_dir)
    data = await read(
        state_no_sim,
        {
            "sources": [{"job_id": job_id}],
            "recipes": [
                {"key": "inj", "metric": "tone", "signal": "V(out)", "frequency_param": "finj"}
            ],
        },
    )
    rows = _rows(data, "inj")
    assert [row["assignments"]["FINJ"] for row in rows] == [finj for finj, _, _ in _CASES]
    for row, (_, amplitude, phase) in zip(rows, _CASES, strict=True):
        assert row["value"]["frequency_hz"] == pytest.approx(
            {"1k": 1e3, "4k": 4e3}.get(row["assignments"]["FINJ"], 2500.0)
        )
        assert row["value"]["amplitude"] == pytest.approx(amplitude, rel=1e-4)
        assert row["value"]["phase_deg"] == pytest.approx(phase, abs=0.01)


@ROUTES
async def test_tone_takes_a_stepped_runs_frequency_from_the_step(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """In a stepped deck the parameter's value is the step's."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "rc"}],
            "recipes": [
                {"key": "t", "metric": "tone", "signal": "V(out)", "frequency_param": "R"}
            ],
            "step": {"axis": "r", "value": 680},
        },
    )
    (row,) = _rows(data, "t")
    assert row["value"]["frequency_hz"] == 680.0


@ROUTES
async def test_tone_refuses_a_run_that_does_not_give_the_parameter(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(_two_tone_raw(work_dir)), "label": "dut"}],
            "recipes": [
                {"key": "t", "metric": "tone", "signal": "V(out)", "frequency_param": "FINJ"}
            ],
        },
    )
    (failure,) = data["failures"]
    assert "'FINJ'" in failure["message"]
    assert "frequency_hz" in failure["message"]


def test_frequency_param_and_frequency_hz_exclude_each_other():
    from ltspice_mcp.lib.recipes import validate_recipe

    base = {"key": "t", "metric": "tone", "signal": "V(out)"}
    with pytest.raises(ValueError, match="frequency_hz"):
        validate_recipe({**base, "frequency_hz": "1k", "frequency_param": "FINJ"})
    with pytest.raises(ValueError, match="frequency_param"):
        validate_recipe(base)


# ---------------------------------------------------------------------------
# tone of a ratio
# ---------------------------------------------------------------------------

_TONE = {"metric": "tone", "frequency_hz": "1k", "window": {"start": "0.37m"}}


@ROUTES
async def test_tone_of_a_ratio_is_the_quotient_of_the_two_tones(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    raw = _two_tone_raw(work_dir)
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [
                {"key": "out", "signal": "V(out)", **_TONE},
                {"key": "in", "signal": "V(in)", **_TONE},
                {"key": "gain", "signal": "V(out)/V(in)", **_TONE},
                {"key": "inverse", "signal": "-1/V(out)", **_TONE},
            ],
        },
    )
    (out,) = (row["value"] for row in _rows(data, "out"))
    (into,) = (row["value"] for row in _rows(data, "in"))
    (gain,) = (row["value"] for row in _rows(data, "gain"))
    (inverse,) = (row["value"] for row in _rows(data, "inverse"))
    quotient = _phasor(out) / _phasor(into)
    assert _phasor(gain) == pytest.approx(quotient, rel=1e-9)
    assert gain["amplitude"] == pytest.approx(0.5, rel=1e-3)
    assert gain["phase_deg"] == pytest.approx(-60.0, abs=0.03)
    assert (gain["t_start"], gain["t_end"], gain["n_cycles"]) == (
        out["t_start"],
        out["t_end"],
        out["n_cycles"],
    )
    assert _phasor(inverse) == pytest.approx(-1 / _phasor(out), rel=1e-9)


@ROUTES
async def test_tone_of_a_ratio_reports_a_denominator_with_no_such_tone(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    t = np.linspace(0.0, 4e-3, 4001)
    raw = work_dir / "flat.raw"
    _write_tran(raw, t, {"V(out)": np.sin(2 * np.pi * 1e3 * t), "V(dc)": np.full_like(t, 2.0)})
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [
                {"key": "g", "metric": "tone", "signal": "V(out)/V(dc)", "frequency_hz": "1k"}
            ],
        },
    )
    (failure,) = data["failures"]
    assert "singular" in failure["message"]
    assert "V(dc)" in failure["message"]


# ---------------------------------------------------------------------------
# The forms an AC recipe reads
# ---------------------------------------------------------------------------


@ROUTES
async def test_the_reciprocal_of_an_ac_response_is_its_reciprocal(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    point = {"metric": "bode_point", "at_hz": "3k"}
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "rc"}],
            "recipes": [
                {"key": "a", "signal": "V(out)", **point},
                {"key": "inv", "signal": "1/V(out)", **point},
                {"key": "neg_inv", "signal": "-1/V(out)", **point},
            ],
            "include": {"fields": ["value"]},
        },
    )
    (a,) = (row["value"]["points"][0] for row in _rows(data, "a"))
    (inv,) = (row["value"]["points"][0] for row in _rows(data, "inv"))
    (neg_inv,) = (row["value"]["points"][0] for row in _rows(data, "neg_inv"))
    assert inv["magnitude_db"] == pytest.approx(-a["magnitude_db"], abs=1e-9)
    assert inv["phase_deg"] == pytest.approx(-a["phase_deg"], abs=1e-9)
    assert neg_inv["magnitude_db"] == pytest.approx(-a["magnitude_db"], abs=1e-9)
    # The same angle as 180 - phase, wrapped into the range the recipe gives.
    turned = math.remainder(neg_inv["phase_deg"] - (180.0 - a["phase_deg"]), 360.0)
    assert turned == pytest.approx(0.0, abs=1e-9)


#: The arguments, beyond its signal, each recipe that reads a signal needs on
#: the recorded RC low-pass (ltspice_ac_rc, 10 Hz to 100 kHz).
_ARGUMENTS: dict[str, dict[str, Any]] = {
    "bode_point": {"at_hz": "3k"},
    "bode_crossing": {"level_db": -3.0},
    "bode_slope": {"from_hz": "10k", "to_hz": "100k"},
    "tone": {"frequency_hz": "1k"},
    "transient_response": {"mode": "step"},
}

#: Every form the reader takes, each spelled a second way that names the same
#: quantity on that run, whose V(in) is exactly 1.
_FORMS = [
    ("-V(out)", "-V(out)/V(in)"),
    ("V(in)/V(out)", "1/V(out)"),
    ("1/V(out)", "V(in)/V(out)"),
    ("-1/V(out)", "-V(in)/V(out)"),
]


def _signal_recipes() -> list[str]:
    return sorted(
        get_args(model.model_fields["metric"].annotation)[0]
        for model in RECIPE_MODELS
        if "signal" in model.model_fields
    )


def _without_signal(entry: dict[str, Any]) -> Any:
    if "failures" in entry:
        return [failure["message"] for failure in entry["failures"]]
    return [
        {key: value for key, value in row["value"].items() if key != "signal"}
        for row in entry["results"]["r"]["values"]
    ]


def _assert_close(first: Any, second: Any, where: Any) -> None:
    if isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys(), where
        for key in first:
            _assert_close(first[key], second[key], (*where, key))
    elif isinstance(first, list):
        assert isinstance(second, list) and len(first) == len(second), where
        for index, (one, other) in enumerate(zip(first, second, strict=True)):
            _assert_close(one, other, (*where, index))
    elif isinstance(first, float):
        assert second == pytest.approx(first, rel=1e-9, abs=1e-9), where
    else:
        assert first == second, where


async def test_every_recipe_that_reads_an_ac_signal_reads_every_form(
    state_no_sim: SessionState, work_dir: Path
):
    """Every recipe that answers on an AC sweep and not on a transient reads
    ``-A``, ``A/B``, ``1/A`` and ``-1/A``, each as the same quantity as its
    other spelling; and those recipes are the ones the guide names."""
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    tran = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    reads_ac = []
    for metric in _signal_recipes():
        recipe = {"key": "r", "metric": metric, **_ARGUMENTS.get(metric, {})}
        on_ac, on_tran = [
            await _through_tool(
                state_no_sim,
                {"sources": [{"path": str(path)}], "recipes": [{**recipe, "signal": "V(out)"}]},
            )
            for path in (raw, tran)
        ]
        if "failures" in on_ac or "failures" not in on_tran:
            continue
        reads_ac.append(metric)
        for form, other in _FORMS:
            first, second = [
                await _through_tool(
                    state_no_sim,
                    {"sources": [{"path": str(raw)}], "recipes": [{**recipe, "signal": signal}]},
                )
                for signal in (form, other)
            ]
            assert "failures" not in first, (metric, form, first["failures"])
            _assert_close(_without_signal(first), _without_signal(second), (metric, form))
    assert reads_ac == [
        "ac_structure",
        "bode_crossing",
        "bode_filter",
        "bode_point",
        "bode_slope",
        "resonance",
        "return_loss",
        "stability",
    ]


def test_the_reader_reads_each_form_and_refuses_the_rest():
    def parts(signal: str) -> tuple[str | None, str | None, bool]:
        expression = metrics.read_signal(signal)
        return expression.numerator, expression.denominator, expression.negated

    assert parts("V(out)") == ("V(out)", None, False)
    assert parts(" - V(out) ") == ("V(out)", None, True)
    assert parts("V(out)/V(in,0)") == ("V(out)", "V(in,0)", False)
    assert parts("-V(out) / V(in)") == ("V(out)", "V(in)", True)
    assert parts("1/V(out)") == (None, "V(out)", False)
    assert parts("-1/gain_1") == (None, "gain_1", True)
    for bad in ("-", "--V(out)", "V(a)/V(b)/V(c)", "V(out)/", "/V(out)", "V(a)/-V(b)", "1/1"):
        with pytest.raises(ResultError, match=r"Signal|ratio"):
            metrics.read_signal(bad)


#: Calls that read an operator in a string: a leading ``-`` or a ``/``.
_OPERATOR_TESTS = re.compile(
    r"""\.(?:startswith|lstrip|removeprefix)\(\s*['"]-['"]"""
    r"""|\.(?:split|partition|rpartition)\(\s*['"]/['"]"""
    r"""|['"][-/]['"]\s+(?:not\s+)?in\s+\w*signal"""
    r"""|\w*signal\w*\[0\]\s*==\s*['"]-['"]""",
)


def test_only_the_reader_looks_for_an_operator_in_a_signal():
    """No code but ``read_signal`` tests a signal for a leading ``-`` or
    splits one at ``/``: a second reader would let one recipe read a form
    another refuses, or read it differently."""
    found = []
    for name, tree in _source_scan.modules().items():
        for scope, nodes in _source_scan.scopes(tree):
            if (name, scope) == ("lib/metrics.py", "read_signal"):
                continue
            for node in nodes:
                if not isinstance(node, ast.Call | ast.Compare):
                    continue
                text = ast.unparse(node)
                if "signal" in text.lower() and _OPERATOR_TESTS.search(text):
                    found.append(f"{name}:{scope}: {text}")
    assert found == []
    assert any(
        _OPERATOR_TESTS.search(ast.unparse(node))
        for scope, nodes in _source_scan.scopes(_source_scan.modules()["lib/metrics.py"])
        if scope == "read_signal"
        for node in nodes
        if isinstance(node, ast.Call)
    ), "the scan no longer recognises the reader's own tests"


# ---------------------------------------------------------------------------
# stability on a frequency response analysis
# ---------------------------------------------------------------------------


def _fra_job(state: SessionState, work_dir: Path, run: str) -> str:
    """A job whose one case ran the recorded frequency response analysis."""
    for suffix in (".raw", ".log", ".fra_1.raw"):
        name = f"{run}{suffix}"
        shutil.copyfile(rec.recorded("ltspice26", f"raw/{name}"), work_dir / name)
    job = make_experiment_job(state, job_id=f"exp_{run}", raw=work_dir / f"{run}.raw")
    job.cases[0].other_raws = ["fra_1"]
    return job.job_id


async def _stability(
    read: Reader, state: SessionState, source: dict[str, Any], signal: str, **extra: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The recipe's rows, and the observations the reply makes of them."""
    data = await read(
        state,
        {
            "sources": [source],
            "recipes": [{"key": "loop", "metric": "stability", "signal": signal}],
            **extra,
        },
    )
    made = [
        observation
        for observation in data.get("observations", [])
        if observation["code"].startswith("fra_margin")
    ]
    return [row["value"] for row in _rows(data, "loop")], made


@ROUTES
async def test_stability_on_the_form_ltspice_measured_says_nothing_more(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    job_id = _fra_job(state_no_sim, work_dir, "fra")
    source = {"job_id": job_id, "other_raw": "fra_1"}
    (value,), made = await _stability(read, state_no_sim, source, "-gain_1")
    assert value["phase_margin_worst_deg"] == pytest.approx(90.6, abs=0.05)
    assert made == []


@pytest.mark.parametrize("signal", ["gain_1", "1/gain_1", "-1/gain_1"])
@ROUTES
async def test_stability_on_another_form_says_its_margin_is_not_ltspices(
    read: Reader, signal: str, state_no_sim: SessionState, work_dir: Path
):
    """LTspice prints the crossover and phase margin of the loop it measured.
    A form whose margin differs from it beyond the printed digits is not that
    loop, and the reply gives both and names the form that agrees."""
    job_id = _fra_job(state_no_sim, work_dir, "fra")
    source = {"job_id": job_id, "other_raw": "fra_1"}
    (value,), (observation,) = await _stability(read, state_no_sim, source, signal)
    assert observation["code"] == "fra_margin_differs"
    assert observation["kind"] == "reconciliation"
    evidence = observation["evidence"]
    assert evidence["recipe"] == "loop"
    assert evidence["rows"] == [{"source": "exp_fra", "case_id": "case-0000", "step_index": 0}]
    assert evidence["printed"] == {
        "analyzer": "@1",
        "crossover_hz": 15900.0,
        "phase_margin_deg": 90.6,
    }
    assert evidence["found"]["phase_margin_deg"] == value["phase_margin_worst_deg"]
    assert evidence["agreeing_forms"] == ["-gain_1"]
    assert "-gain_1" in observation["detail"]
    assert "90.6" in observation["detail"]


@ROUTES
async def test_a_stepped_fra_run_is_checked_step_by_step(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    job_id = _fra_job(state_no_sim, work_dir, "fra_step")
    source = {"job_id": job_id, "other_raw": "fra_1"}
    right, right_made = await _stability(read, state_no_sim, source, "-gain_1", all_steps=True)
    wrong, wrong_made = await _stability(read, state_no_sim, source, "gain_1", all_steps=True)
    assert (len(right), len(wrong), right_made) == (2, 2, [])
    assert [observation["code"] for observation in wrong_made] == ["fra_margin_differs"] * 2
    assert [observation["evidence"]["rows"][0]["step_index"] for observation in wrong_made] == [
        0,
        1,
    ]
    assert [observation["evidence"]["printed"] for observation in wrong_made] == [
        {"analyzer": "@1", "crossover_hz": 15900.4, "phase_margin_deg": 90.6295},
        {"analyzer": "@1", "crossover_hz": 8007.62, "phase_margin_deg": 90.1991},
    ]


@ROUTES
async def test_stability_on_fra_results_with_no_printed_margin_says_it_did_not_check(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """Read by its path, the results file has no log beside it to compare with."""
    raw = work_dir / "fra.fra_1.raw"
    shutil.copyfile(rec.recorded("ltspice26", "raw/fra.fra_1.raw"), raw)
    _, (observation,) = await _stability(read, state_no_sim, {"path": str(raw)}, "-gain_1")
    assert observation["code"] == "fra_margin_not_checked"
    assert observation["kind"] == "coverage"
    assert "not established" in observation["detail"]


async def test_stability_on_an_ac_sweep_compares_with_nothing(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    (_,), made = await _stability(_through_tool, state_no_sim, {"path": str(raw)}, "-V(out)")
    assert made == []
