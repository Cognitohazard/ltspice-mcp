"""Summary consumes captured resident log facts without reopening inputs."""

from pathlib import Path

import numpy as np
import pytest
from spicelib import RawRead

from ltspice_mcp.lib import log_parser
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.log_decode import LogLimits, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from ltspice_mcp.lib.raw_parser import build_simulation_summary
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
    raw = make_raw_mock(
        ["V(out)"] if operating else ["time", "V(out)"],
        axis,
        {"time": axis, "V(out)": values},
        plotname="Operating Point" if operating else "Transient Analysis",
        steps=steps,
    )
    if operating:
        raw.get_axis.side_effect = RuntimeError("This RAW file does not have an axis.")
    return raw


def test_recorded_summary_uses_facts_after_all_files_removed(tmp_path, monkeypatch):
    raw = RawRead(str(FIXTURES_DIR / "ltspice_tran_rc.raw"), dialect="ltspice")
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
    raw = RawRead(str(FIXTURES_DIR / f"{name}.raw"), dialect="ltspice")
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
        "Circuit: test\n.step R=1\n.step R=2\ngmin stepping failed\n",
        "Circuit: test\nDirect Newton iteration succeeded in finding operating point.\n"
        "Direct Newton iteration failed to find operating point.\ngmin stepping failed\n",
    ],
)
def test_op_attempts_or_step_rows_keep_later_failure(tmp_path, body):
    summary = build_simulation_summary(
        _raw(operating=True), captured_log_facts(tmp_path, text=body)
    )
    assert summary["errors"] == ["gmin stepping failed"]
    warning = next(w for w in summary["warnings"] if "Stepped .op detected" in w)
    assert "2 bias-point iterations" in warning
    assert ".dc R" in warning if ".step R" in body else ".dc <param>" in warning


@pytest.mark.parametrize(
    ("values", "steps", "recovered"),
    [
        ([1, 1, 1], None, True),
        ([1e30, 1, 1], None, False),
        ([float("nan"), 1, 1], None, False),
        ([1, 1, 1], [0, 1], False),
    ],
)
def test_single_solve_recovery_requires_finite_complete_raw(tmp_path, values, steps, recovered):
    log = captured_log_facts(tmp_path, text="Circuit: test\ngmin stepping failed\n")
    summary = build_simulation_summary(_raw(values=values, steps=steps), log)
    assert ("errors" not in summary) is recovered
    assert any("OP solve recovered" in w for w in summary.get("warnings", [])) is recovered


@pytest.mark.parametrize("section", ["steps", "op_iterations"])
def test_unknown_op_coverage_never_demotes_solve_error(tmp_path, section):
    metadata = captured_log_facts(tmp_path, text="Circuit: test\ngmin stepping failed\n").as_dict()
    metadata[section] = {
        "status": "error",
        "value": None,
        "nonfinite_count": 0,
        "error": {"type": "ValueError", "message": "synthetic unknown solve coverage"},
    }
    summary = build_simulation_summary(_raw(), DecodedLog(metadata))
    assert summary["errors"] == ["gmin stepping failed"]
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
    log = captured_log_facts(
        tmp_path,
        text=(
            "Circuit: test\nexample.cir(9): No such function defined.\n"
            ".meas AC fc_3dB WHEN vdb(out)=-3\n^^^\n"
        ),
    )
    summary = build_simulation_summary(_raw(), log)
    assert summary["meas_errors"] == log.value("diagnostics")["meas_errors"]
    entry = summary["meas_errors"][0]
    assert entry["directive"] == ".meas AC fc_3dB WHEN vdb(out)=-3"
    assert "^^^" in entry["raw_block"]
    assert "mag" in entry["suggestion"].lower()
    assert any(o["code"] == "meas_parse_error" for o in summary["observations"])


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
