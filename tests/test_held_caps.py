"""A value above a cap is served at the cap and said so, never refused.

A cap bounds what one call costs, so asking for more of the same thing gets
the most the call will serve and a note naming the cap. The hygiene check
finds every tool-input field that refuses a value above a bound (``le=`` or
``lt=``, on the field or on the items it holds) and requires each to be a bound
on what the value means rather than on what the call costs; the rest send a
value over each cap through its tool and read the note.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin

import annotated_types
import numpy as np
import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from ltspice_mcp import api
from ltspice_mcp.lib.model_fields import models_in
from ltspice_mcp.lib.recipes import WAVEFORM_MAX_POINTS_CAP
from ltspice_mcp.lib.signal_analysis import THD_HARMONICS_CAP
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analysis, inspect_tools, run_code
from ltspice_mcp.tools._base import RENDER_SCALE_CAP, registry
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit
from tests.conftest import FIXTURES_DIR, needs_raster, ngspice_binary_raw

#: The input fields that refuse a value above their bound, each with why the
#: bound is on what the value means, so holding it would answer another question.
_REFUSING_ALLOWED: dict[str, str] = {
    "SignalStatsRecipe.quantiles": "a quantile is a fraction of the samples, at most 1",
    "PdkNativeVariation.seed": "the largest root seed the sampler takes; another is another sample",
    "ExperimentExecution.simulator_seed": (
        "the largest seed ngspice takes; another seed is another run"
    ),
}


def _every_input_model() -> set[type[BaseModel]]:
    import ltspice_mcp.tools  # noqa: F401  (registers every tool)

    pending: list[type[BaseModel]] = [
        r.input_model
        for r in registry._registered  # pyright: ignore[reportPrivateUsage]
        if r.input_model is not None
    ]
    seen: set[type[BaseModel]] = set()
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending += list(model.__dict__.get("VARIANTS", ()))
        for field in model.model_fields.values():
            pending += models_in(field.annotation)
    return seen


def _constraints(annotation: Any) -> list[Any]:
    """The constraints ``Annotated`` puts anywhere in ``annotation``: on it, on a
    union's member, on a list's items."""
    found: list[Any] = []
    if get_origin(annotation) is Annotated:
        for meta in annotation.__metadata__:
            found += meta.metadata if isinstance(meta, FieldInfo) else [meta]
    for argument in get_args(annotation):
        found += _constraints(argument)
    return found


def _refusing_above(models: set[type[BaseModel]]) -> set[str]:
    """``Model.field`` for every field of ``models`` that refuses a value above
    a bound (``le=`` or ``lt=``), on the field itself or on what it holds."""
    return {
        f"{model.__name__}.{name}"
        for model in models
        for name, field in model.model_fields.items()
        if any(
            isinstance(meta, annotated_types.Le | annotated_types.Lt)
            for meta in [*field.metadata, *_constraints(field.annotation)]
        )
    }


def test_no_capped_argument_refuses_a_value_above_its_cap() -> None:
    models = _every_input_model()
    assert len(models) >= 50, "the walk found too few input models to be checking anything"
    refusing = _refusing_above(models)
    unexplained = sorted(refusing - set(_REFUSING_ALLOWED))
    assert not unexplained, (
        "these refuse a value above their bound instead of holding it with a note "
        "(NotedModel.hold_to_cap), and are not listed as a bound on meaning: "
        + ", ".join(unexplained)
    )
    stale = sorted(set(_REFUSING_ALLOWED) - refusing)
    assert not stale, "listed fields that no longer refuse above a bound: " + ", ".join(stale)


def test_the_refusing_bound_rule_catches_what_it_names() -> None:
    class Planted(BaseModel):
        limit: int = Field(default=10, le=100)
        wait_s: float | None = Field(default=None, lt=60)
        depth: Annotated[int, Field(le=8)] | None = None
        weights: list[Annotated[float, Field(le=1)]] = []
        nested: dict[str, list[Annotated[int, Field(lt=4)]]] = {}
        count: int = Field(default=1, ge=1)
        names: list[str] = Field(default_factory=list, max_length=4)

    assert _refusing_above({Planted}) == {
        "Planted.limit",
        "Planted.wait_s",
        "Planted.depth",
        "Planted.weights",
        "Planted.nested",
    }


def _note(name: str, asked: int | float, cap: int | float, unit: str = "") -> str:
    def spelled(value: int | float) -> str:
        return f"{value:d}" if isinstance(value, int) else f"{value:g}"

    return (
        f"{name}={spelled(asked)}{unit} is above its cap of {spelled(cap)}{unit}; "
        f"{spelled(cap)}{unit} was used."
    )


def _raw(work_dir: Path) -> Path:
    raw = work_dir / "rc.raw"
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.log", work_dir / "rc.log")
    return raw


async def _inspect(state: SessionState, query: dict[str, Any]) -> dict[str, Any]:
    args = inspect_tools.InspectInput.model_validate({"queries": [query]})
    data = (await inspect_tools.handle_inspect(args, state)).structured_content
    assert data is not None
    (item,) = data["results"]
    return item


class TestInspect:
    async def test_a_results_limit_above_its_cap_is_held(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        item = await _inspect(
            state_no_sim, {"kind": "results", "view": "signals", "path": str(raw), "limit": 500}
        )
        assert item["ok"], item
        assert item["data"]["warnings"] == [_note("limit", 500, inspect_tools.RESULTS_LIMIT_CAP)]

    async def test_a_reference_limit_above_its_cap_is_held(
        self, state_no_sim: SessionState
    ) -> None:
        item = await _inspect(state_no_sim, {"kind": "reference", "query": "thd", "limit": 500})
        assert item["ok"], item
        assert item["data"]["warnings"] == [_note("limit", 500, inspect_tools.REFERENCE_LIMIT_CAP)]

    async def test_a_limit_within_its_cap_says_nothing(self, state_no_sim: SessionState) -> None:
        item = await _inspect(state_no_sim, {"kind": "reference", "query": "thd", "limit": 3})
        assert item["ok"] and "warnings" not in item["data"]


class _RecordingWorker:
    """Stands in for the snippet worker process: records the bound it is given."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    async def run(self, code: str, timeout_s: float, reset: bool) -> dict[str, Any]:
        self.timeouts.append(timeout_s)
        return {
            "status": "ok",
            "exec_seq": 1,
            "worker_pid": None,
            "stdout": "",
            "stderr": "",
            "result": None,
            "truncated": False,
            "chars_dropped": 0,
            "elapsed_s": 0.0,
            "error": None,
            "running": None,
            "worker_restarted": None,
            "hint": None,
        }


class TestRunCode:
    async def test_a_timeout_above_its_cap_is_held(self, state_no_sim: SessionState) -> None:
        worker = _RecordingWorker()
        state_no_sim.code_worker = worker  # type: ignore[assignment]
        args = run_code.RunCodeInput.model_validate({"code": "1", "timeout_s": 1000})
        result = await run_code.handle_run_code(args, state_no_sim)
        data = result.structured_content
        assert data is not None
        assert worker.timeouts == [run_code.RUN_CODE_TIMEOUT_CAP_S]
        note = _note("timeout_s", 1000.0, run_code.RUN_CODE_TIMEOUT_CAP_S, "s")
        assert data["warnings"] == [note]
        assert data["hint"].startswith(note)


class TestPlotWaveform:
    async def test_max_points_above_its_cap_is_held_and_said(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _raw(work_dir)
        asked = WAVEFORM_MAX_POINTS_CAP + 1_000_000
        args = analysis.PlotWaveformInput.model_validate(
            {"path": str(raw), "signals": ["V(out)"], "max_points": asked, "open": False}
        )
        data = (await analysis.handle_plot_waveform(args, state_no_sim)).structured_content
        assert data is not None
        assert data["max_points"] == WAVEFORM_MAX_POINTS_CAP
        assert data["warnings"] == [_note("max_points", asked, WAVEFORM_MAX_POINTS_CAP)]


class TestWaveformRecipe:
    @pytest.mark.parametrize("asked", [50_000, WAVEFORM_MAX_POINTS_CAP + 1])
    async def test_max_points_above_the_served_cap_is_held_and_said(
        self, state_no_sim: SessionState, work_dir: Path, asked: int
    ) -> None:
        raw = _raw(work_dir)
        cap = min(WAVEFORM_MAX_POINTS_CAP, state_no_sim.config.max_points_returned)
        args = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "rc"}],
                "recipes": [
                    {"key": "w", "metric": "waveform", "signals": ["V(out)"], "max_points": asked}
                ],
            }
        )
        data = (await handle_analyze_results(args, state_no_sim)).structured_content
        assert data is not None
        entry = data["results"]["w"]
        assert entry["warnings"] == [_note("max_points", asked, cap)]


def _tone() -> list[tuple[float, float]]:
    """A 1 kHz tone with a tenth of its second harmonic, ten periods."""
    return [
        (t, math.sin(2 * math.pi * 1e3 * t) + 0.1 * math.sin(2 * math.pi * 2e3 * t))
        for t in (n / 200e3 for n in range(2000))
    ]


def _sine_raw(work_dir: Path) -> Path:
    """The tone, written as ngspice writes a transient plot and read by the real parser."""
    rows = _tone()
    raw = work_dir / "tone.raw"
    raw.write_bytes(ngspice_binary_raw(rows, ["time", "v(out)"], declared=len(rows)))
    return raw


class TestThdRecipe:
    async def _thd(self, state: SessionState, raw: Path, harmonics: int) -> dict[str, Any]:
        args = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "tone", "dialect": "ngspice"}],
                "recipes": [
                    {
                        "key": "d",
                        "metric": "thd",
                        "signal": "v(out)",
                        "fundamental_hz": "1k",
                        "harmonics": harmonics,
                    }
                ],
            }
        )
        data = (await handle_analyze_results(args, state)).structured_content
        assert data is not None
        return data["results"]["d"]

    async def test_harmonics_above_the_cap_are_summed_to_it_and_said(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        raw = _sine_raw(work_dir)
        asked = THD_HARMONICS_CAP + 10
        held = await self._thd(state_no_sim, raw, asked)
        at_cap = await self._thd(state_no_sim, raw, THD_HARMONICS_CAP)
        assert held["warnings"] == [_note("harmonics", asked, THD_HARMONICS_CAP)]
        assert "warnings" not in at_cap
        assert held["values"] == at_cap["values"]
        (row,) = held["values"]
        # Harmonics 2 to the cap, the fundamental being the first.
        assert row["value"]["n_harmonics_used"] == THD_HARMONICS_CAP - 1
        assert row["value"]["thd_pct"] == pytest.approx(10.0, rel=1e-3)

    def test_the_python_primitive_sums_to_the_same_cap_and_says_so(self) -> None:
        """api.analyze_thd answers the same request the recipe does, under its
        own argument's name."""
        t, y = (np.array(column) for column in zip(*_tone(), strict=True))
        asked = THD_HARMONICS_CAP + 10
        held = api.analyze_thd(t, y, fundamental=1e3, n_harmonics=asked)
        at_cap = api.analyze_thd(t, y, fundamental=1e3, n_harmonics=THD_HARMONICS_CAP)
        assert held["warnings"] == [_note("n_harmonics", asked, THD_HARMONICS_CAP)]
        assert at_cap["warnings"] == []
        assert {**held, "warnings": []} == at_cap
        assert held["n_harmonics_used"] == THD_HARMONICS_CAP - 1


_SHEET = (
    "Version 4.1\nSHEET 1 880 680\nSYMBOL res 400 300 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\n"
)


async def _verify(state: SessionState, work_dir: Path, render: dict[str, Any]) -> dict[str, Any]:
    sheet = work_dir / "r.asc"
    sheet.write_text(_SHEET, encoding="utf-8", newline="\n")
    args = VerifyCircuitInput.model_validate({"path": str(sheet), "render": render})
    data = (await handle_verify_circuit(args, state)).structured_content
    assert data is not None
    return data


@pytest.mark.usefixtures("asc_symbols")
class TestVerifyCircuit:
    async def test_a_render_scale_above_its_cap_is_held_and_said(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        data = await _verify(state_no_sim, work_dir, {"mode": "only", "format": "svg", "scale": 8})
        assert "render" in data, data
        note = "render." + _note("scale", 8.0, RENDER_SCALE_CAP)
        assert data["warnings"] == [note]
        assert data["hint"].startswith(note.rstrip("."))

    @needs_raster
    async def test_the_png_is_drawn_at_the_cap(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        data = await _verify(state_no_sim, work_dir, {"mode": "only", "scale": 8})
        assert data["render"]["scale"] == RENDER_SCALE_CAP
        assert data["warnings"] == ["render." + _note("scale", 8.0, RENDER_SCALE_CAP)]

    async def test_a_scale_within_its_cap_says_nothing(
        self, state_no_sim: SessionState, work_dir: Path
    ) -> None:
        data = await _verify(state_no_sim, work_dir, {"mode": "only", "format": "svg", "scale": 2})
        assert "render" in data and "warnings" not in data
