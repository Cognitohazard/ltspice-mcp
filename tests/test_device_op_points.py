"""Device operating-point egress: LTspice logopinfo log block + auto-injection.

Covers the two pure halves of reading per-device op points (gm/gds/vth/vdsat)
off LTspice: parsing the log's "Semiconductor Device Operating Points" block
(read_device_op_points) and adding ``.options logopinfo`` to LTspice ``.op``
decks so the block is produced (inject_logopinfo). The fold into
operating_point is covered in test_analysis_tools.py.
"""

from pathlib import Path

from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.lib.log_parser import read_device_op_points
from ltspice_mcp.tools._base import inject_logopinfo

# A trimmed real LTspice .op log (LTspice 26.x) with the logopinfo block.
_LOG_WITH_BLOCK = """\
LTspice 26.0.2 for Windows
Direct Newton iteration succeeded in finding operating point.
Semiconductor Device Operating Points:
                        --- MOSFET Transistors ---
Name:           M1
Model:         nch
Id:          9.60e-05
Vgs:         9.00e-01
Vds:         1.80e+00
Vth:         5.00e-01
Vdsat:       4.00e-01
Gm:          4.80e-04
Gds:         1.00e-06
Total elapsed time: 0.067 seconds.
"""


class TestReadDeviceOpPoints:
    def test_parses_block_keyed_at_dev_param(self, tmp_path: Path):
        log = tmp_path / "op.log"
        log.write_text(_LOG_WITH_BLOCK)
        params = read_device_op_points(log)
        assert params["@m1[gm]"] == 4.8e-4
        assert params["@m1[vth]"] == 0.5
        assert params["@m1[vdsat]"] == 0.4
        assert params["@m1[id]"] == 9.6e-5
        # The string ``Model:`` row is not a number, so it is dropped.
        assert "@m1[model]" not in params

    def test_no_block_returns_empty(self, tmp_path: Path):
        log = tmp_path / "plain.log"
        log.write_text("Direct Newton iteration succeeded.\nTotal elapsed time: 0.01 s.\n")
        assert read_device_op_points(log) == {}

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert read_device_op_points(tmp_path / "nope.log") == {}


class _NotLTspice:
    """Stand-in for a non-LTspice simulator class."""


class TestInjectLogopinfo:
    _DECK = (
        "* cs stage\n"
        "M1 d g 0 0 nch L=1u W=10u\n"
        "Vd d 0 1.8\n"
        ".model nch nmos (level=1 vto=0.5 kp=120u)\n"
        ".op\n"
        ".end\n"
    )

    def _write(self, tmp_path: Path, text: str, name: str = "cs.cir") -> Path:
        p = tmp_path / name
        p.write_text(text)
        return p

    def test_injects_before_end_for_ltspice_op(self, tmp_path: Path):
        src = self._write(tmp_path, self._DECK)
        run = inject_logopinfo(src, LTspice, "job1")
        assert run != src
        out = run.read_text()
        assert ".options logopinfo" in out
        # Inserted before the final .end, and the original deck is untouched.
        assert out.index("logopinfo") < out.index(".end")
        assert src.read_text() == self._DECK

    def test_per_job_unique_names_avoid_clobber(self, tmp_path: Path):
        # Two queued/concurrent runs of the SAME netlist must not share an
        # augmented file — otherwise a later run overwrites the deck the earlier
        # queued job has yet to read. The job_id stamp keeps them distinct.
        src = self._write(tmp_path, self._DECK)
        a = inject_logopinfo(src, LTspice, "jobA")
        b = inject_logopinfo(src, LTspice, "jobB")
        assert a != b
        assert a.exists() and b.exists()
        assert "jobA" in a.name and "jobB" in b.name

    def test_noop_when_already_present(self, tmp_path: Path):
        deck = self._DECK.replace(".op\n", ".options logopinfo\n.op\n")
        src = self._write(tmp_path, deck)
        assert inject_logopinfo(src, LTspice, "job1") == src

    def test_noop_without_op_directive(self, tmp_path: Path):
        # .tran (no .op) produces no op-point block, so don't inject.
        deck = self._DECK.replace(".op\n", ".tran 1m\n")
        src = self._write(tmp_path, deck)
        assert inject_logopinfo(src, LTspice, "job1") == src

    def test_dot_options_alone_does_not_trigger(self, tmp_path: Path):
        # ``.op\b`` must not match ``.options`` — a deck with .options but no .op
        # is not an op-point run.
        deck = self._DECK.replace(".op\n", ".options reltol=1e-4\n.tran 1m\n")
        src = self._write(tmp_path, deck)
        assert inject_logopinfo(src, LTspice, "job1") == src

    def test_noop_for_non_ltspice(self, tmp_path: Path):
        src = self._write(tmp_path, self._DECK)
        assert inject_logopinfo(src, _NotLTspice, "job1") == src

    def test_injects_with_existing_other_options(self, tmp_path: Path):
        deck = self._DECK.replace(".op\n", ".options reltol=1e-4\n.op\n")
        src = self._write(tmp_path, deck)
        run = inject_logopinfo(src, LTspice, "job1")
        assert run != src
        assert ".options logopinfo" in run.read_text()
