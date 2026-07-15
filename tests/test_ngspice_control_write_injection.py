"""ngspice ``.control``-block write injection.

A ``.control`` block replaces ngspice's default raw output (ngspice runtime
behavior, not a spicelib bug — see ``inject_ngspice_control_write``'s
docstring): the script runs instead of the default post-analysis raw write,
so a deck that never calls ``write``/``wrdata`` itself produces no raw for
the analysis tools to read even though the run completed cleanly.
``inject_ngspice_control_write`` adds a canonical ``write <rawpath>`` before
``.endc`` so that raw exists at the path the runner already expects for the
job. This file covers the pure transform + its guards; the live-ngspice
integration test (run_simulation on a real ``.control`` deck) lives in
test_ngspice_e2e.py, gated on ngspice being on PATH.
"""

from pathlib import Path

from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.tools._base import inject_ngspice_control_write


class _NotNgspice:
    """Stand-in for a non-ngspice simulator class."""


class TestInjectNgspiceControlWrite:
    _DECK = (
        "* rc step\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\n"
        "R1 in out 1k\n"
        "C1 out 0 1u\n"
        ".tran 1u 5m\n"
        ".control\n"
        "run\n"
        ".endc\n"
        ".end\n"
    )

    def _write(self, tmp_path: Path, text: str, name: str = "ctrl.cir") -> Path:
        p = tmp_path / name
        p.write_text(text)
        return p

    def _out_dir(self, tmp_path: Path) -> Path:
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        return out_dir

    def test_injects_write_before_endc(self, tmp_path: Path):
        src = self._write(tmp_path, self._DECK)
        out_dir = self._out_dir(tmp_path)
        run = inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir)
        assert run != src
        text = run.read_text()
        assert "write " in text
        # Inserted inside the block, before .endc, and the original is untouched.
        assert text.index("write ") < text.index(".endc")
        assert src.read_text() == self._DECK

    def test_write_target_is_absolute_job_raw_path_unquoted(self, tmp_path: Path):
        # Must match the exact path spicelib's own (suppressed) -r would use,
        # and must be absolute: the runner passes no cwd, so ngspice inherits
        # the server's own working directory, not the output folder. Written
        # UNQUOTED — ngspice's `write` parser can't handle a quoted target
        # (verified empirically: quoting or escaping both fail).
        src = self._write(tmp_path, self._DECK)
        out_dir = self._out_dir(tmp_path)
        run = inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir)
        expected = (out_dir / "job1.raw").as_posix()
        text = run.read_text()
        assert f"write {expected}\n" in text
        assert f'"{expected}"' not in text

    def test_noop_without_control_block(self, tmp_path: Path):
        deck = self._DECK.replace(".control\nrun\n.endc\n", "")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_noop_when_write_already_present(self, tmp_path: Path):
        # Never override a user who already captures their own output.
        deck = self._DECK.replace("run\n", "run\nwrite myresults.raw\n")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_noop_when_wrdata_already_present(self, tmp_path: Path):
        deck = self._DECK.replace("run\n", "run\nwrdata out.txt v(out)\n")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_noop_for_non_ngspice(self, tmp_path: Path):
        # LTspice has no .control; inject_logopinfo covers its own injection.
        src = self._write(tmp_path, self._DECK)
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, _NotNgspice, "job1", out_dir) == src

    def test_noop_for_unsupported_extension(self, tmp_path: Path):
        src = self._write(tmp_path, self._DECK, name="ctrl.asc")
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_noop_when_output_folder_has_a_space(self, tmp_path: Path):
        # ngspice's `write` parser can't handle a spaced target at all — not
        # quoted, not escaped — so a spaced output folder must skip the
        # injection rather than emit a write ngspice would choke on; the run
        # falls back to today's harmless log-only behavior.
        src = self._write(tmp_path, self._DECK)
        out_dir = tmp_path / "run dir"
        out_dir.mkdir()
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_noop_with_two_control_blocks(self, tmp_path: Path):
        # Ambiguous which block's result is "the" answer — skip rather than guess.
        deck = self._DECK + ".control\nrun\n.endc\n"
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        assert inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir) == src

    def test_inserted_before_trailing_quit(self, tmp_path: Path):
        # quit ends control-script execution — a write placed after it would
        # never run, so it must land before the LAST quit/exit in the block.
        deck = self._DECK.replace(".endc", "quit\n.endc")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        run = inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir)
        text = run.read_text()
        assert text.index("write ") < text.index("quit")

    def test_conditional_quit_does_not_anchor_the_write(self, tmp_path: Path):
        # A quit nested in an if is NOT the block's script-ending statement.
        # Anchoring on it would bury the write inside the conditional, so it
        # would never run on the normal (condition-false) path — the exact bug
        # this feature exists to prevent. The write must land before .endc,
        # i.e. AFTER the conditional quit, not before it.
        deck = self._DECK.replace(".endc", "if $foo\nquit\nend\n.endc")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        run = inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir)
        text = run.read_text()
        assert text.index("write ") > text.index("quit")
        assert text.index("write ") < text.index(".endc")

    def test_case_insensitive_control_and_endc(self, tmp_path: Path):
        deck = self._DECK.replace(".control", ".CONTROL").replace(".endc", ".ENDC")
        src = self._write(tmp_path, deck)
        out_dir = self._out_dir(tmp_path)
        run = inject_ngspice_control_write(src, NGspiceSimulator, "job1", out_dir)
        assert run != src
        assert "write " in run.read_text()

    def test_per_job_unique_names_avoid_clobber(self, tmp_path: Path):
        # Two queued/concurrent runs of the SAME netlist must not share an
        # augmented file, or a later run overwrites the deck an earlier
        # queued job has yet to read.
        src = self._write(tmp_path, self._DECK)
        out_dir = self._out_dir(tmp_path)
        a = inject_ngspice_control_write(src, NGspiceSimulator, "jobA", out_dir)
        b = inject_ngspice_control_write(src, NGspiceSimulator, "jobB", out_dir)
        assert a != b
        assert a.exists() and b.exists()
        assert "jobA" in a.name and "jobB" in b.name
