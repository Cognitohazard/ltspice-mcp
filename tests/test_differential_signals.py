"""A node-pair voltage ``V(a,b)`` reads as ``V(a) - V(b)`` on every signal reader.

No simulator writes a ``V(a,b)`` trace, so the shared resolver builds it from
the two node voltages the raw does carry: one raw, one step, one axis. Every
test here runs a recorded raw through a real reader and checks the answer
against the two operand traces read on their own.
"""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from ltspice_mcp.api import RawResult
from ltspice_mcp.api._primitives import load_raw_result
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.recipes import RECIPE_MODELS
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests.conftest import (
    SyncApi,
    call_tool_params,
    fake_request_context,
    stage_recorded_fixture,
    tool_text,
)


async def _analyze(
    state: SessionState, raw: Path, recipes: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    args = AnalyzeResultsInput.model_validate(
        {"sources": [{"raw_path": str(raw), "label": "dut"}], "recipes": recipes, **extra}
    )
    result = await handle_analyze_results(args, state)
    assert result.structured_content is not None
    return result.structured_content


async def _load(state: SessionState, path: Path) -> RawResult:
    """The raw as ``api.load_raw`` hands it back, awaited on the test's own loop."""
    return await load_raw_result(
        state=state, raw_path=path, job_id=None, run_index=0, case_id=None
    )


def _value(data: dict[str, Any], key: str) -> dict[str, Any]:
    assert data["failures"] == []
    return data["results"][key]["values"][0]["value"]


def _failure_message(data: dict[str, Any]) -> str:
    assert data["results"] == {} or all(not r.get("values") for r in data["results"].values())
    assert len(data["failures"]) == 1
    return data["failures"][0]["message"]


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _plot_data(path: Path) -> dict[str, Any]:
    """The plot spec a rendered page embeds."""
    html = path.read_text(encoding="utf-8")
    start = html.index('type="application/json">') + len('type="application/json">')
    return json.loads(html[start : html.index("</script>", start)].replace("<\\/", "</"))


def _nearest(axis: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(axis - x)))


# ---------------------------------------------------------------------------
# RawResult.trace — the Python API reads the same resolver
# ---------------------------------------------------------------------------


class TestRawResultTrace:
    def test_differential_is_the_per_step_difference(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = SyncApi(state_no_sim).load_raw(stage_recorded_fixture(work_dir, "ltspice_step_tran"))
        assert raw.step_count == 3
        for step in range(raw.step_count):
            expected = raw.trace("V(a)", step=step) - raw.trace("V(out)", step=step)
            got = raw.trace("V(a,out)", step=step)
            assert got.shape == raw.axis(step=step).shape
            np.testing.assert_array_equal(got, expected)
        # Different steps carry different data, so a step mix-up cannot pass.
        assert not np.array_equal(raw.trace("V(a,out)", step=0), raw.trace("V(a,out)", step=2))

    def test_ac_differential_subtracts_complex_values(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = SyncApi(state_no_sim).load_raw(stage_recorded_fixture(work_dir, "ltspice_ac_rc"))
        got = raw.trace("V(in,out)")
        assert np.iscomplexobj(got)
        np.testing.assert_array_equal(got, raw.trace("V(in)") - raw.trace("V(out)"))
        # V(in) - V(out) is the drop across R1; its phase is not V(in)'s phase
        # minus a magnitude, which is what a real-part-only subtraction gives.
        assert np.any(np.abs(np.imag(got)) > 0)

    @pytest.mark.parametrize("spelling", ["V(out,0)", "V(out,gnd)", "v( OUT , GND )"])
    def test_ground_operand_reads_as_the_node_voltage(
        self, state_no_sim: SessionState, work_dir: Path, spelling: str
    ):
        raw = SyncApi(state_no_sim).load_raw(stage_recorded_fixture(work_dir, "ltspice_tran_rc"))
        np.testing.assert_array_equal(raw.trace(spelling), raw.trace("V(out)"))

    def test_missing_operand_is_named(self, state_no_sim: SessionState, work_dir: Path):
        raw = SyncApi(state_no_sim).load_raw(stage_recorded_fixture(work_dir, "ltspice_tran_rc"))
        with pytest.raises(ResultError, match=r"V\(nowhere\)") as excinfo:
            raw.trace("V(in,nowhere)")
        # The operand that exists is not blamed.
        assert "'V(in)'" not in str(excinfo.value).split("Available")[0]

    def test_noise_densities_do_not_subtract(self, state_no_sim: SessionState, work_dir: Path):
        raw = SyncApi(state_no_sim).load_raw(stage_recorded_fixture(work_dir, "ltspice_noise_rc"))
        with pytest.raises(ResultError, match="spectral densit"):
            raw.trace("V(onoise,inoise)")


# ---------------------------------------------------------------------------
# analyze_results recipes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecipes:
    async def test_value_on_a_transient(self, state_no_sim: SessionState, work_dir: Path):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        raw = await _load(state_no_sim, path)
        axis = raw.axis()
        index = _nearest(axis, 900e-6)
        expected = float(raw.trace("V(in)")[index] - raw.trace("V(out)")[index])

        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "drop", "metric": "value", "expr": "V(in,out)", "at": "900u"}],
        )
        value = _value(data, "drop")
        assert value["signal"] == "V(in,out)"
        assert value["value"] == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert value["actual_x"] == pytest.approx(float(axis[index]))
        assert value["unit"] == "V"

    async def test_value_reads_each_step_on_its_own_axis(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        raw = await _load(state_no_sim, path)
        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "v", "metric": "value", "expr": "V(a,out)", "at": "900u"}],
            all_steps=True,
            include={"per_run": {"limit": 10}},
        )
        assert data["failures"] == []
        rows = data["results"]["v"]["per_run"]["items"]
        assert [row["step_index"] for row in rows] == [0, 1, 2]
        for row in rows:
            step = row["step_index"]
            axis = raw.axis(step=step)
            index = _nearest(axis, 900e-6)
            expected = raw.trace("V(a)", step=step)[index] - raw.trace("V(out)", step=step)[index]
            assert row["value"]["value"] == pytest.approx(float(expected), rel=1e-12, abs=1e-15)

    async def test_value_on_an_ac_run_is_the_complex_difference(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        raw = await _load(state_no_sim, path)
        freq = float(raw.axis()[40])
        expected = complex(raw.trace("V(in)")[40] - raw.trace("V(out)")[40])

        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "v", "metric": "value", "expr": "V(in,out)", "at": repr(freq)}],
        )
        value = _value(data, "v")
        assert value["magnitude_linear"] == pytest.approx(abs(expected), rel=1e-9)
        assert value["phase_deg"] == pytest.approx(np.angle(expected, deg=True), abs=1e-9)

    async def test_signal_stats(self, state_no_sim: SessionState, work_dir: Path):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        raw = await _load(state_no_sim, path)
        diff = raw.trace("V(in)") - raw.trace("V(out)")
        data = await _analyze(
            state_no_sim, path, [{"key": "s", "metric": "signal_stats", "signal": "V(in,out)"}]
        )
        stats = _value(data, "s")
        assert stats["max"] == pytest.approx(float(np.max(diff)))
        assert stats["min"] == pytest.approx(float(np.min(diff)))

    async def test_ratio_over_a_differential_denominator(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Across an RC low-pass, V(out)/V(in,out) = 1/(sRC): an integrator whose
        # phase is -90 degrees everywhere and whose unity-gain frequency is the
        # low-pass corner. A wrong subtraction cannot produce both.
        path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        raw = await _load(state_no_sim, path)
        index = 70  # a decade above the corner, where |H| is far from 0 dB
        freq = float(raw.axis()[index])
        h = complex(
            raw.trace("V(out)")[index] / (raw.trace("V(in)")[index] - raw.trace("V(out)")[index])
        )

        data = await _analyze(
            state_no_sim,
            path,
            [
                {"key": "corner", "metric": "bode_filter", "signal": "V(out)"},
                {"key": "loop", "metric": "stability", "signal": "V(out)/V(in,out)"},
                {
                    "key": "point",
                    "metric": "bode_point",
                    "signal": "V(out)/V(in,out)",
                    "at_hz": freq,
                },
            ],
        )
        assert data["failures"] == []
        corner = data["results"]["corner"]["values"][0]["value"]
        loop = data["results"]["loop"]["values"][0]["value"]
        point = data["results"]["point"]["values"][0]["value"]
        assert loop["unity_gain_hz"] == pytest.approx(corner["cutoff_high_hz"], rel=0.02)
        assert loop["phase_margin_worst_deg"] == pytest.approx(90.0, abs=0.5)
        assert point["signal"] == "V(out)/V(in,out)"
        assert 20 * np.log10(abs(h)) < -10
        assert point["magnitude_db"] == pytest.approx(20 * np.log10(abs(h)), abs=1e-6)

    async def test_inline_waveform(self, state_no_sim: SessionState, work_dir: Path):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        raw = await _load(state_no_sim, path)
        data = await _analyze(
            state_no_sim, path, [{"key": "w", "metric": "waveform", "signals": ["V(in,out)"]}]
        )
        series = _value(data, "w")["series"][0]
        assert series["signal"] == "V(in,out)"
        np.testing.assert_allclose(series["y"], raw.trace("V(in)") - raw.trace("V(out)"))

    async def test_csv_waveform(self, state_no_sim: SessionState, work_dir: Path):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        raw = await _load(state_no_sim, path)
        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "w", "metric": "waveform", "signals": ["V(in,out)"], "format": "csv"}],
        )
        rows = await asyncio.to_thread(_csv_rows, Path(_value(data, "w")["artifact"]["path"]))
        assert rows[0][-1] == "V(in,out)"
        column = [float(row[-1]) for row in rows[1:]]
        np.testing.assert_allclose(column, raw.trace("V(in)") - raw.trace("V(out)"))

    async def test_value_at_a_bias_point(self, state_no_sim: SessionState, work_dir: Path):
        raw = work_dir / "pair_op.raw"
        raw.write_text(
            "Title: * operating point\n"
            "Plotname: Operating Point\nFlags: real\n"
            "No. Variables: 3\nNo. Points: 1\n"
            "Command: Linear Technology Corporation LTspice\nVariables:\n"
            "\t0\tV(inp)\tvoltage\n"
            "\t1\tV(inn)\tvoltage\n"
            "\t2\tI(R1)\tdevice_current\n"
            "Values:\n0\t1.25\n\t1.2497\n\t0.001\n",
            encoding="utf-8",
            newline="\n",
        )
        data = await _analyze(
            state_no_sim,
            raw,
            [
                {"key": "offset", "metric": "value", "expr": "V(inp,inn)"},
                {"key": "inp", "metric": "value", "expr": "V(inp)"},
                {"key": "inn", "metric": "value", "expr": "V(inn)"},
            ],
        )
        value = _value(data, "offset")
        assert value["signal"] == "V(inp,inn)"
        # The raw stores single precision; the pair is exactly the difference of
        # the two node voltages this same recipe reads.
        assert value["value"] == _value(data, "inp")["value"] - _value(data, "inn")["value"]
        assert value["value"] == pytest.approx(3e-4, rel=1e-3)
        assert value["unit"] == "V"

    async def test_missing_operand_fails_naming_it(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "v", "metric": "value", "expr": "V(nowhere,out)", "at": "900u"}],
        )
        assert "V(nowhere)" in _failure_message(data)


# ---------------------------------------------------------------------------
# plot_waveform
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plot_waveform_draws_the_difference(state_no_sim: SessionState, work_dir: Path):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw = await _load(state_no_sim, path)
    result = await handle_plot_waveform(
        PlotWaveformInput(raw_file=str(path), signals=["V(in,out)"], open=False), state_no_sim
    )
    data = result.structured_content
    assert data is not None
    assert data["signals"] == ["V(in,out)"]
    blob = await asyncio.to_thread(_plot_data, Path(data["path"]))
    (panel,) = blob["panels"]
    assert panel["series"] == [{"label": "V(in,out)"}]
    np.testing.assert_allclose(panel["data"][1], raw.trace("V(in)") - raw.trace("V(out)"))


# ---------------------------------------------------------------------------
# Trace math that is not a node pair points at Python
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExpressionPointer:
    async def test_analyze_results_names_run_code(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        assert "run_code" in state_no_sim.tool_dispatch
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim,
            path,
            [{"key": "v", "metric": "value", "expr": "V(in)-V(out)", "at": "900u"}],
        )
        message = _failure_message(data)
        assert "run_code" in message
        assert "r.trace(" in message
        assert "api.load_raw(" in message

    async def test_names_the_library_when_run_code_is_off(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        state = SessionState.create(replace(state_no_sim.config, run_code=False), available={})
        assert "run_code" not in state.tool_dispatch
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state, path, [{"key": "s", "metric": "signal_stats", "signal": "2*V(out)"}]
        )
        message = _failure_message(data)
        assert "run_code" not in message
        assert "from ltspice_mcp.api import Api" in message
        assert "r.trace(" in message

    async def test_tool_error_names_run_code(self, state_no_sim: SessionState, work_dir: Path):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await call_tool(
            fake_request_context(state_no_sim),
            call_tool_params(
                "plot_waveform",
                {"raw_file": str(path), "signals": ["abs(V(out))"], "open": False},
            ),
        )
        assert result.is_error
        text = tool_text(result)
        assert "run_code" in text
        assert "r.trace(" in text
        # The generic "check the job" hint would misdirect here.
        assert 'jobs (action:"status")' not in text

    @pytest.mark.parametrize("name", ["V(in-)", "V(nowhere)", "I(Q1)"])
    async def test_a_plain_missing_trace_gets_no_pointer(
        self, state_no_sim: SessionState, work_dir: Path, name: str
    ):
        # A sign inside the parentheses is part of a node name, not an operator.
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim, path, [{"key": "v", "metric": "value", "expr": name, "at": "900u"}]
        )
        message = _failure_message(data)
        assert "not found" in message
        assert "run_code" not in message
        assert "r.trace(" not in message

    async def test_python_api_error_carries_the_snippet(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = await _load(state_no_sim, stage_recorded_fixture(work_dir, "ltspice_tran_rc"))
        with pytest.raises(ResultError, match=r"r\.trace\(") as excinfo:
            raw.trace("V(in)-V(out)")
        # A library caller is already in Python; the tool is not named to it.
        assert "run_code" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Every signal-taking recipe reads through the resolver
# ---------------------------------------------------------------------------

#: Each recipe that names a signal, on a raw of the run type it reads.
SIGNAL_RECIPES = [
    ("value", "ltspice_tran_rc", {"expr": "V(out)", "at": "900u"}),
    ("signal_stats", "ltspice_tran_rc", {"signal": "V(out)"}),
    ("edges", "ltspice_tran_rc", {"signal": "V(out)"}),
    ("timing", "ltspice_tran_rc", {"from": {"signal": "V(in)"}, "to": {"signal": "V(out)"}}),
    ("periodic", "ltspice_step_tran", {"signal": "V(out)"}),
    ("transient_response", "ltspice_tran_rc", {"signal": "V(out)", "mode": "step"}),
    (
        "transient_response",
        "ltspice_tran_rc",
        {"signal": "V(out)", "mode": "disturbance", "input": "V(in)"},
    ),
    ("thd", "ltspice_step_tran", {"signal": "V(out)"}),
    ("bode_filter", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("bode_point", "ltspice_ac_rc", {"signal": "V(out)", "at_hz": "1k"}),
    ("bode_crossing", "ltspice_ac_rc", {"signal": "V(out)", "level_db": -3.0}),
    ("bode_slope", "ltspice_ac_rc", {"signal": "V(out)", "from_hz": "10k", "to_hz": "100k"}),
    ("stability", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("ac_structure", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("resonance", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("return_loss", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("waveform", "ltspice_tran_rc", {"signals": ["V(out)"], "max_points": 25}),
    ("waveform", "ltspice_ac_rc", {"signals": ["V(out)"], "max_points": 25}),
    ("plot", "ltspice_tran_rc", {"signals": ["V(out)"]}),
    ("noise_integral", "ltspice_noise_rc", {"signal": "V(onoise)"}),
]

#: The fields through which a recipe names a signal.
_SIGNAL_FIELDS = {"signal", "signals", "expr", "input", "from_", "to"}


def test_the_sweep_names_every_recipe_that_takes_a_signal():
    # A recipe added with a signal field and left out of the list above would
    # never be shown to read a node pair; this is what notices.
    takes_signal = {
        get_args(model.model_fields["metric"].annotation)[0]
        for model in RECIPE_MODELS
        if _SIGNAL_FIELDS & set(model.model_fields)
    }
    assert takes_signal == {metric for metric, _, _ in SIGNAL_RECIPES}


def _respell(fields: Any, spellings: dict[str, str]) -> Any:
    if isinstance(fields, str):
        return spellings.get(fields, fields)
    if isinstance(fields, list):
        return [_respell(item, spellings) for item in fields]
    if isinstance(fields, dict):
        return {key: _respell(item, spellings) for key, item in fields.items()}
    return fields


@pytest.mark.asyncio
@pytest.mark.parametrize(("metric", "fixture_name", "fields"), SIGNAL_RECIPES)
async def test_a_ground_operand_answers_exactly_like_the_trace(
    state_no_sim: SessionState, work_dir: Path, metric: str, fixture_name: str, fields: dict
):
    path = stage_recorded_fixture(work_dir, fixture_name)
    grounded = _respell(
        fields, {"V(out)": "V(out,0)", "V(in)": "V(in,gnd)", "V(onoise)": "V(onoise,0)"}
    )
    data = await _analyze(
        state_no_sim,
        path,
        [
            {"key": "trace", "metric": metric, **fields},
            {"key": "pair", "metric": metric, **grounded},
        ],
    )
    assert data["failures"] == []
    # Some metrics echo the signal as it was asked for, and a chart lands at a
    # path of its own; every number agrees.
    echoes = {"signal", "signal_a", "signal_b", "artifact"}
    pair = {k: v for k, v in _value(data, "pair").items() if k not in echoes}
    assert pair == {k: v for k, v in _value(data, "trace").items() if k not in echoes}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metric", "fixture_name", "fields"),
    [case for case in SIGNAL_RECIPES if case[0] != "noise_integral"],
)
async def test_a_node_pair_runs_on_every_signal_recipe(
    state_no_sim: SessionState, work_dir: Path, metric: str, fixture_name: str, fields: dict
):
    path = stage_recorded_fixture(work_dir, fixture_name)
    pair = _respell(fields, {"V(out)": "V(in,out)"})
    data = await _analyze(state_no_sim, path, [{"key": "pair", "metric": metric, **pair}])
    assert data["failures"] == []


@pytest.mark.asyncio
async def test_noise_integral_refuses_a_genuine_pair(state_no_sim: SessionState, work_dir: Path):
    path = stage_recorded_fixture(work_dir, "ltspice_noise_rc")
    data = await _analyze(
        state_no_sim, path, [{"key": "n", "metric": "noise_integral", "signal": "V(onoise,R1)"}]
    )
    assert "spectral densities" in _failure_message(data)
