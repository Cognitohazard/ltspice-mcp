"""Whether an AC plot's frequency axis is real, on every path that reads it.

ngspice 42 writes the frequency of every AC plot as a complex number whose
imaginary part is a subnormal number, about 4.6e-310, at every point, in binary
and in ASCII raw files (``fixtures/ngspice_ac_rerun``, written by ngspice 42).
No simulator computes such a coordinate, so the axis is its real part, as
LTspice's is (every recorded LTspice AC and FRA axis holds an imaginary part
of exactly 0). An axis whose imaginary part can be a coordinate is still
refused, with what reads the plot instead.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import raw_parser, result_kinds
from ltspice_mcp.lib.ac_analysis import prepare_ac_arrays
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, SyncApi

_RERUN = FIXTURES_DIR / "ngspice_ac_rerun"
_FREQUENCIES = [1e3 * 10 ** (k / 5) for k in range(16)]


async def _every_reading(state: SessionState, raw: Path) -> dict[str, Any]:
    """The file read through every path that reads an AC plot's axis: the
    decoder, plot_waveform, analyze_results (a value at a frequency, a filter
    reading and the summary), inspect and the Python API."""
    readings: dict[str, Any] = {}
    api = SyncApi(state)
    loaded = await asyncio.to_thread(api.load_raw, str(raw), dialect="ngspice")
    axis = loaded._raw.descriptor.axis
    assert axis is not None
    readings["real_coordinates"] = axis.real_coordinates
    try:
        readings["api"] = loaded.axis()
    except ResultError as refused:
        readings["api"] = str(refused)
    try:
        plotted = await handle_plot_waveform(
            PlotWaveformInput.model_validate(
                {"path": str(raw), "dialect": "ngspice", "signals": ["v(out)"], "open": False}
            ),
            state,
        )
    except ResultError as refused:
        readings["plot"] = str(refused)
    else:
        assert plotted.structured_content is not None
        readings["plot"] = plotted.structured_content["traces"]
    analyzed = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "dialect": "ngspice", "label": "ac"}],
                "recipes": [
                    {"key": "at", "metric": "value", "expr": "v(out)", "at": "10k"},
                    {"key": "filter", "metric": "bode_filter", "signal": "v(out)"},
                ],
            }
        ),
        state,
    )
    assert analyzed.structured_content is not None, analyzed.content
    readings["analyzed"] = analyzed.structured_content
    summarized = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "dialect": "ngspice", "label": "ac"}],
                "recipes": [{"key": "s", "metric": "summary"}],
                "include": {"per_run": {"limit": 10}, "fields": ["value"]},
            }
        ),
        state,
    )
    assert summarized.structured_content is not None, summarized.content
    (row,) = summarized.structured_content["results"]["s"]["per_run"]["items"]
    readings["summary"] = row["value"]
    inspected = await handle_inspect(
        InspectInput.model_validate(
            {
                "queries": [
                    {"kind": "results", "view": "signals", "path": str(raw), "dialect": "ngspice"}
                ]
            }
        ),
        state,
    )
    assert inspected.structured_content is not None
    (readings["inspect"],) = inspected.structured_content["results"]
    return readings


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["a.raw", "a_ascii.raw", "b.raw", "b_ascii.raw"])
async def test_ngspice_42s_ac_axis_reads_as_its_frequencies_on_every_path(
    name: str, state_no_sim: SessionState, work_dir: Path
):
    raw = work_dir / name
    shutil.copy(_RERUN / name, raw)
    readings = await _every_reading(state_no_sim, raw)
    assert readings["real_coordinates"] is True
    np.testing.assert_allclose(readings["api"], _FREQUENCIES, rtol=1e-12)
    (trace,) = readings["plot"]
    assert trace["signal"] == "v(out)"
    results = readings["analyzed"]["results"]
    assert "error" not in results["at"], results["at"]
    assert "error" not in results["filter"], results["filter"]
    assert "failures" not in readings["analyzed"], readings["analyzed"]["failures"]
    summary = readings["summary"]
    assert summary["range"]["freq_start"] == pytest.approx(1e3)
    assert summary["range"]["freq_end"] == pytest.approx(_FREQUENCIES[-1])
    assert "ac_bandwidth_metrics" in summary
    assert readings["inspect"]["ok"], readings["inspect"]["error"]["message"]


def _truly_complex(tmp_path: Path) -> Path:
    """ngspice 42's ASCII file with the imaginary part of the fourth frequency
    one that can be a coordinate."""
    text = (_RERUN / "a_ascii.raw").read_text(encoding="ascii")
    lines = text.split("\n")
    (index,) = [i for i, line in enumerate(lines) if line.startswith("3\t\t")]
    lines[index] = re.sub(r",\S+$", ",1.000000000000000e+02", lines[index])
    path = tmp_path / "complex_axis.raw"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


@pytest.mark.asyncio
async def test_an_axis_whose_imaginary_part_can_be_a_coordinate_is_refused_on_every_path(
    state_no_sim: SessionState, work_dir: Path
):
    readings = await _every_reading(state_no_sim, _truly_complex(work_dir))
    assert readings["real_coordinates"] is False
    assert "non-real coordinates" in readings["api"]
    assert "non-real coordinates" in readings["plot"]
    failures = readings["analyzed"]["failures"]
    assert sorted(f["where"] for f in failures) == ["ac:0", "ac:0"]
    for failure in failures:
        assert "non-real coordinates" in failure["message"], failure
    summary = readings["summary"]
    assert not summary.get("range")
    assert any("non-real coordinates" in warning for warning in summary["warnings"])
    bandwidth = summary["ac_bandwidth_metrics"]
    assert bandwidth.get("bandwidth_3db") is None
    assert any("non-real coordinates" in warning for warning in bandwidth["warnings"])


def test_the_rule_sets_aside_only_an_imaginary_part_that_cannot_have_been_data():
    tiny = np.finfo(np.float64).tiny
    stray = np.array([1e3 + 4.6e-310j, 0 + 4.6e-310j, 2e3 + 0j])
    np.testing.assert_array_equal(result_kinds.axis_coordinates(stray), [1e3, 0.0, 2e3])
    for imaginary in (tiny, 1e-300, 1e-12, 0.5, np.nan, np.inf):
        assert result_kinds.axis_coordinates(np.array([1e3 + 0j, 2e3 + imaginary * 1j])) is None
    real = np.array([1.0, 2.0])
    assert result_kinds.axis_coordinates(real) is real
    # Every reader that is handed such an axis refuses it rather than reading
    # its real part.
    complex_axis = np.array([1e3 + 0j, 2e3 + 0.5j, 3e3 + 0j])
    with pytest.raises(ResultError, match="non-real coordinates"):
        raw_parser.real_axis(complex_axis)
    with pytest.raises(ValueError, match="non-real coordinates"):
        prepare_ac_arrays(complex_axis, np.ones(3, dtype=complex))
    freqs, _ = prepare_ac_arrays(np.real(complex_axis) + 4.6e-310j, np.ones(3, dtype=complex))
    np.testing.assert_array_equal(freqs, [1e3, 2e3, 3e3])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id",
    ["raw/ac", "raw/ac_as_text", "raw/step_ac", "raw/fra.fra_1", "raw/fra_step.fra_1"],
)
async def test_a_recorded_ltspice_frequency_axis_holds_no_imaginary_part(
    case_id: str, tmp_path: Path
):
    """The rule leaves LTspice's reading as it was: each recorded AC and FRA
    axis holds an imaginary part of exactly 0 at every point."""
    for build in rec.BUILDS:
        if not rec.recorded(build, f"{case_id}.raw").is_file():
            continue
        raw = rec.decode(build, case_id, tmp_path / build).raw
        assert raw.descriptor.axis is not None
        assert raw.descriptor.axis.real_coordinates
        for step in range(max(1, len(raw.descriptor.steps))):
            axis = np.asarray(raw.get_axis(step=step))
            assert np.iscomplexobj(axis)
            assert np.all(np.imag(axis) == 0)


def test_two_reruns_whose_axes_differ_only_in_a_coordinate_are_not_the_same_result(
    tmp_path: Path,
):
    """The result digest sets aside only an imaginary part that cannot be a
    coordinate: ngspice 42's reruns digest alike, a changed coordinate does not."""
    first = raw_parser.result_digest(_RERUN / "a_ascii.raw", "ngspice")
    assert first is not None
    assert raw_parser.result_digest(_RERUN / "b_ascii.raw", "ngspice") == first
    assert raw_parser.result_digest(_truly_complex(tmp_path), "ngspice") != first


def _last_point_imaginary(path: Path, out: Path, imaginary: float) -> Path:
    """``path`` with the imaginary part of its last frequency set to ``imaginary``."""
    data = path.read_bytes()
    if b"Binary:\n" in data:
        record = 16 * 4  # the frequency and three complex values
        at = len(data) - record + 8
        data = data[:at] + struct.pack("<d", imaginary) + data[at + 8 :]
    else:
        lines = data.decode("ascii").split("\n")
        (index,) = [i for i, line in enumerate(lines) if line.startswith("15\t\t")]
        lines[index] = re.sub(r",\S+$", f",{imaginary:.15e}", lines[index])
        data = "\n".join(lines).encode("ascii")
    out.write_bytes(data)
    return out


@pytest.mark.parametrize("name", ["a.raw", "a_ascii.raw"])
def test_a_runs_progress_names_its_last_frequency_only_when_the_axis_is_real(
    name: str, tmp_path: Path
):
    """The progress of a run (``read_partial_raw_progress``) reads the last
    frequency as every reader reads the axis: ngspice 42's as its real part,
    and none when its imaginary part can be a coordinate."""
    progress = raw_parser.read_partial_raw_progress(_RERUN / name, "ngspice")
    assert progress is not None
    assert progress.last_axis_value == pytest.approx(1e6)
    changed = _last_point_imaginary(_RERUN / name, tmp_path / name, 3.0)
    progress = raw_parser.read_partial_raw_progress(changed, "ngspice")
    assert progress is not None
    assert progress.points == 16
    assert progress.last_axis_value is None
