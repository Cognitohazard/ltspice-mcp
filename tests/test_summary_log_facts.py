"""Summary consumes captured resident log facts without reopening inputs."""

from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib import log_parser
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.log_decode import LogLimits, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from ltspice_mcp.lib.raw_parser import build_simulation_summary
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, LTSPICE_TRAN_RC_VFINAL, make_raw_mock
from tests.test_log_decode import fourier_text


def captured_log_facts(
    tmp_path: Path,
    source: Path | None = None,
    *,
    text: str | None = None,
    console: str | None = None,
) -> DecodedLog:
    """Use actual bounded capture and worker decode for legacy summary tests."""
    directory = tmp_path / f"summary-worker-{len(list(tmp_path.glob('summary-worker-*')))}"
    directory.mkdir()
    if text is not None:
        source = directory / "source.log"
        source.write_text(text, encoding="utf-8")
    console_path = None
    if console is not None:
        console_path = directory / "source.exe.log"
        console_path.write_text(console, encoding="utf-8")
    limits = LogLimits(2_000_000, 16_000, 20_000, 50_000, 2_000_000)
    captured = capture_inputs(
        SourceFiles(
            raw=FIXTURES_DIR / "ltspice_tran_rc.raw"
            if source is None and console is None
            else None,
            log=source,
            console=console_path,
        ),
        directory,
        input_bytes=limits.log_bytes,
        log_bytes=limits.log_bytes,
    )
    return DecodedLog(decode_logs(captured, directory, limits=limits))


def _raw(*, operating=False, values=None, steps=None):
    values = np.array([1.0, 1.0, 1.0]) if values is None else np.array(values)
    axis = np.arange(len(values), dtype=float)
    return make_raw_mock(
        ["V(out)"] if operating else ["time", "V(out)"],
        axis,
        {"time": axis, "V(out)": values},
        plotname="Operating Point" if operating else "Transient Analysis",
        steps=steps,
    )


def test_recorded_summary_uses_facts_after_all_files_removed(tmp_path, monkeypatch):
    raw = rec.decode_file(FIXTURES_DIR / "ltspice_tran_rc.raw", tmp_path).raw
    source = tmp_path / "recorded.log"
    source.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.log").read_bytes())
    log = captured_log_facts(tmp_path, source)
    source.unlink()
    for directory in tmp_path.glob("summary-worker-*"):
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Summary reopened files or constructed a dependency log reader")

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(log_parser, "make_log_reader", forbidden)
        guarded.setattr(log_parser, "LTSpiceLogReader", forbidden)
        summary = build_simulation_summary(raw, log, duration=1.25, value_scan=True)
    assert summary["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]
    assert summary["measurements"]["vfinal"]["at"] == 0.0009
    assert summary["temp_c"] == summary["tnom_c"] == 27.0
    assert summary["duration"] == 1.25
    assert summary["observations"] == []
    assert "warnings" not in summary


@pytest.mark.parametrize("name", ["ltspice_ac_rc", "ltspice_dc_div", "op_extreme_node"])
def test_recorded_summary_log_facts_have_no_invented_results(tmp_path, name):
    raw = rec.decode_file(FIXTURES_DIR / f"{name}.raw", tmp_path).raw
    source = FIXTURES_DIR / f"{name}.log"
    log = captured_log_facts(tmp_path, source if source.exists() else None)
    summary = build_simulation_summary(raw, log)
    assert "measurements" not in summary
    assert "fourier" not in summary
    assert "errors" not in summary


def test_fourier_measurements_and_copy_isolation(tmp_path):
    log = captured_log_facts(tmp_path, text=fourier_text())
    before = log.as_dict()
    summary = build_simulation_summary(_raw(), log)
    assert summary["measurements"]["vrms"]["values"] == [0.5]
    entry = summary["fourier"][0]
    assert entry["thd"] == 0.014047
    assert entry["thd_unit"] == "%"
    assert entry["fundamental_frequency"] == 1000.0
    assert entry["harmonics"][0]["magnitude"] == 0.8464
    entry["harmonics"][0]["magnitude"] = 99
    summary["measurements"]["vrms"]["values"][0] = 99
    assert log.as_dict() == before


@pytest.mark.parametrize("console", [None, "Error: analysis not run\n"])
def test_absent_log_and_console_only_facts(tmp_path, console):
    log = captured_log_facts(tmp_path, console=console)
    summary = build_simulation_summary(_raw(), log)
    assert "measurements" not in summary and "fourier" not in summary
    assert "temp_c" not in summary
    if console:
        assert summary["errors"] == ["Error: analysis not run"]
        assert summary["observations"][0]["kind"] == "relay"
    else:
        assert "warnings" not in summary and summary["observations"] == []


@pytest.mark.parametrize(
    ("section", "label"),
    [
        ("temperatures", "temperatures"),
        ("measurements", "measurements"),
        ("diagnostics", "log diagnostics"),
        ("fourier", "fourier"),
        ("steps", "step rows"),
        ("op_iterations", "OP iterations"),
    ],
)
def test_section_errors_surface_original_exception(tmp_path, section, label):
    metadata = captured_log_facts(tmp_path, text="Circuit: test\n").as_dict()
    metadata[section] = {
        "status": "error",
        "value": None,
        "nonfinite_count": 0,
        "error": {"type": "ValueError", "message": "synthetic section corruption"},
    }
    summary = build_simulation_summary(_raw(), DecodedLog(metadata))
    assert f"{label}" in " ".join(summary["warnings"])
    assert "ValueError: synthetic section corruption" in " ".join(summary["warnings"])


@pytest.mark.parametrize(
    "body",
    [
        "Circuit: test\n.step R=1\n.step R=2\nGmin stepping failed to find operating point.\n",
        "Circuit: test\nDirect Newton iteration succeeded in finding operating point.\n"
        "Direct Newton iteration failed to find operating point.\n"
        "Gmin stepping failed to find operating point.\n",
    ],
)
def test_op_attempts_or_step_rows_say_the_op_is_stepped(tmp_path, body):
    """A stepped ``.op`` says so; its stepping rung, which LTspice writes with
    no error word, is no error (``log_conditions.said_as``)."""
    summary = build_simulation_summary(
        _raw(operating=True), captured_log_facts(tmp_path, text=body)
    )
    assert "errors" not in summary
    warning = next(w for w in summary["warnings"] if "Stepped .op detected" in w)
    assert "2 bias-point iterations" in warning
    assert ".dc R" in warning if ".step R" in body else ".dc <param>" in warning


#: A rung of the bias-point search written with an error word: no recorded
#: build writes one, so it is injected into the decoded log, for the demote
#: that acts on the condition's code.
_ERROR_RUNG = {
    "code": "operating_point_fallback",
    "count": 1,
    "first_text": "Error: gmin stepping failed",
    "channel": "log",
    "severity": "Error",
}


def _with_error_rung(tmp_path, **sections) -> DecodedLog:
    metadata = captured_log_facts(tmp_path, text="Circuit: test\n").as_dict()
    metadata["diagnostics"]["value"]["conditions"] = {"wording": [dict(_ERROR_RUNG)]}
    metadata.update(sections)
    return DecodedLog(metadata)


@pytest.mark.parametrize(
    ("values", "steps", "recovered"),
    [
        ([1, 1, 1], None, True),
        ([1e30, 1, 1], None, False),
        ([float("nan"), 1, 1], None, False),
        ([1, 1, 1], [0, 1], False),
    ],
)
def test_an_error_rung_is_demoted_by_its_code_where_the_raw_holds_the_point(
    tmp_path, values, steps, recovered
):
    """An ``operating_point_fallback`` condition the simulator called an
    error is demoted to a warning when one solve wrote finite, complete node
    data: the point was found after all. It is read by the condition's code,
    never by its wording."""
    summary = build_simulation_summary(
        _raw(values=values, steps=steps), _with_error_rung(tmp_path)
    )
    assert ("errors" not in summary) is recovered
    assert any("OP solve recovered" in w for w in summary.get("warnings", [])) is recovered


@pytest.mark.parametrize("section", ["steps", "op_iterations"])
def test_unknown_op_coverage_never_demotes_solve_error(tmp_path, section):
    unknown = {
        "status": "error",
        "value": None,
        "nonfinite_count": 0,
        "error": {"type": "ValueError", "message": "synthetic unknown solve coverage"},
    }
    summary = build_simulation_summary(_raw(), _with_error_rung(tmp_path, **{section: unknown}))
    assert summary["errors"] == ["Error: gmin stepping failed"]
    assert "unknown solve coverage" in " ".join(summary["warnings"])


def test_nonfinite_log_numbers_remain_null_and_are_explicit(tmp_path):
    body = fourier_text("nan", "inf") + "bad: V(out)=nan\ntemp = 1e999\n"
    summary = build_simulation_summary(_raw(), captured_log_facts(tmp_path, text=body))
    assert summary["measurements"]["bad"]["values"] == [None]
    assert summary["fourier"][0]["thd"] is None
    assert "temp_c" not in summary
    for section in ("temperatures", "measurements", "fourier"):
        assert any(section in w and "non-finite" in w for w in summary["warnings"])


def test_failed_meas_and_diagnostics_reach_reconciliation(tmp_path):
    log = captured_log_facts(
        tmp_path,
        text=('Circuit: test\nMeasurement "late" FAIL\'ed\nWarning: node out is floating\n'),
    )
    summary = build_simulation_summary(_raw(), log, requested={"meas": ["late"], "four": []})
    assert summary["failed_measurements"] == ["late"]
    assert "Warning: node out is floating" in summary["warnings"]
    assert any(o["evidence"].get("reason") == "failed" for o in summary["observations"])


def test_structured_meas_error_preserves_suggestion_and_raw_block(tmp_path):
    """A ``.meas`` the simulator called an error (XVII's wording) is a
    ``meas_errors`` entry with the fix the pre-flight rules suggest, and is
    relayed with the simulator's own word for it."""
    log = captured_log_facts(
        tmp_path,
        text=("Circuit: test\n.meas AC fc_3dB WHEN vdb(out)=-3\nError: unknown function vdb\n"),
    )
    summary = build_simulation_summary(_raw(), log)
    (entry,) = summary["meas_errors"]
    assert entry["directive"] == ".meas AC fc_3dB WHEN vdb(out)=-3"
    assert "Error: unknown function vdb" in entry["raw_block"]
    assert "mag" in entry["suggestion"].lower()
    (relayed,) = [o for o in summary["observations"] if o["code"] == "meas_parse_error"]
    assert relayed["severity"] == "Error"


def test_a_meas_error_with_no_severity_word_is_no_meas_error(tmp_path):
    """LTspice 26 writes a ``.meas`` it cannot read as ``file(N): message``
    with no error word: it is in the summary's diagnostics only."""
    log = captured_log_facts(
        tmp_path,
        text=(
            "Circuit: test\nexample.cir(9): No such function defined.\n"
            ".meas AC fc_3dB WHEN vdb(out)=-3\n^^^\n"
        ),
    )
    summary = build_simulation_summary(_raw(), log)
    assert "meas_errors" not in summary
    assert "errors" not in summary


def test_resident_log_facts_preserve_axisless_first_trace_value_observations(tmp_path):
    log = captured_log_facts(
        tmp_path, text=("Circuit: test\nWarning: node out is floating\nError: unresolved node\n")
    )
    summary = build_simulation_summary(
        _raw(operating=True, values=[1e9, float("nan")]),
        log,
        value_scan=True,
    )
    assert summary["range"] == {}
    assert any(o["code"] == "extreme_value" for o in summary["observations"])
    assert any(o["code"] == "non_finite" for o in summary["observations"])
    assert any(o["kind"] == "relay" for o in summary["observations"])
