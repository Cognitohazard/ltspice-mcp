"""Regression: ngspice prints some analysis diagnostics to stdout while exiting
0 and writing only results to the ``-o`` log. When the runner captures that
stream (``exe_log=True``) into a sibling ``.exe.log``, extract_log_diagnostics
must fold it in so the diagnostics surface in observations.
"""

from pathlib import Path

from ltspice_mcp.lib.log_parser import (
    extract_log_diagnostics,
    make_log_reader,
    parse_measurements,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_exe_log_stdout_diagnostics_surface(tmp_path: Path):
    # The -o log looks clean...
    log = tmp_path / "sim_x.log"
    log.write_text("Circuit: test\n")
    # ...but ngspice printed a convergence failure to stdout, captured here.
    log.with_suffix(".exe.log").write_text(
        "ngspice-44 done\ndoAnalyses: iteration limit reached\n"
    )

    diag = extract_log_diagnostics(log)
    assert any("iteration limit reached" in e.lower() for e in diag["errors"])


def test_missing_exe_log_leaves_result_unchanged(tmp_path: Path):
    # Defensive-read regression: an absent sibling .exe.log is a no-op.
    log = tmp_path / "sim_y.log"
    log.write_text("Circuit: test\n")  # no diagnostics, no sibling exe.log

    diag = extract_log_diagnostics(log)
    assert diag["errors"] == []
    assert diag["warnings"] == []


def test_ngspice_title_echo_not_surfaced_as_measurement():
    """A real ngspice log with NO .meas must not yield a phantom 'circuit'
    measurement.

    ngspice echoes the deck title on a ``Circuit: <title>`` line; spicelib's
    LTspice-shaped reader misparses it as a measurement named 'circuit', scraping
    the first number from the title text (here 1591.55) as its value. The fixture
    is a recorded real ngspice log so a hand-built dict can't paper over the seam.
    """
    log = FIXTURES / "ngspice_ac_no_meas.log"
    # The artifact is real at the spicelib layer: the reader does surface 'circuit'.
    reader = make_log_reader(log)
    assert "circuit" in {n.lower() for n in reader.get_measure_names()}
    # ...but parse_measurements must filter it — the deck declares no real .meas.
    out = parse_measurements(log, reader=reader)
    assert "circuit" not in {k.lower() for k in out["measurements"]}
    assert out["measurements"] == {}
