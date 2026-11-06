"""Reading the compatibility mode a run's log names, and asking ngspice which
flags it gives a mode (``lib/ngspice_mode.py``).

The end-to-end check, a run read in another mode failing, is in
``tests/test_ngspice_startup_mode.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib import ngspice_mode
from ltspice_mcp.lib.ngspice_mode import check_run, flags_for, logged_flags

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

# The head of a log ngspice-42 wrote for an ordinary run.
ORDINARY = "\nNote: Compatibility modes selected: ps lt ki a\n\n\nCircuit: * divider\n\n"
# A driver that sets hsa and sources the deck: the driver is read first.
DRIVER = (
    "\nNote: Compatibility modes selected: ps lt ki a\n\n\nCircuit: * driver\n\n"
    "\nNote: Compatibility modes selected: hs a\n\n\nCircuit: * m multiplier\n\n"
)


@pytest.mark.parametrize(
    ("text", "flags"),
    [
        (ORDINARY, ("ps", "lt", "ki", "a")),
        (ORDINARY.replace("\n", "\r\n"), ("ps", "lt", "ki", "a")),
        (DRIVER, ("hs", "a")),
        ("\nNote: No compatibility mode selected!\n\n\nCircuit: * r\n", ()),
        ("\r\nNote: No compatibility mode selected!\r\n\r\nCircuit: * r\r\n", ()),
        ("\nCircuit: * made in mc mode\n\nDoing analysis at TEMP = 27\n", None),
        ("", None),
    ],
    ids=["ordinary", "crlf", "driver-last", "no-mode", "no-mode-crlf", "no-note", "empty"],
)
def test_the_last_note_names_the_mode_of_the_deck(text: str, flags: tuple[str, ...] | None):
    assert logged_flags(text) == flags


def test_a_note_quoted_inside_a_line_is_not_a_note():
    assert logged_flags("* title Note: Compatibility modes selected: hs a\n") is None


def test_a_missing_log_leaves_the_run_unchecked(tmp_path: Path):
    for log in (None, tmp_path / "absent.log"):
        check = check_run(log, NGspiceSimulator, "kiltpsa")
        assert check.status == "unchecked"
        assert check.reason
        assert check.evidence() == {"ngbehavior": "kiltpsa"}


def test_a_simulator_that_cannot_be_asked_leaves_the_run_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ngspice_mode, "_flags_by_mode", {})
    log = tmp_path / "run.log"
    log.write_text(ORDINARY, encoding="utf-8")

    class Missing(NGspiceSimulator):
        spice_exe: ClassVar[list[str]] = [str(tmp_path / "no-such-ngspice")]

    check = check_run(log, Missing, "kiltpsa")
    assert check.status == "unchecked"
    assert check.logged == ("ps", "lt", "ki", "a")
    # A failure to ask is not kept, so a later run asks again.
    assert ngspice_mode._flags_by_mode == {}


@needs_ngspice
@pytest.mark.parametrize(
    ("mode", "flags"),
    [("", ()), ("hsa", ("hs", "a")), ("kiltpsa", ("ps", "lt", "ki", "a"))],
)
def test_ngspice_is_asked_with_no_startup_file(
    mode: str, flags: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A ``spinit`` and a ``.spiceinit`` that set another mode do not reach
    the answer, or a run they replaced the mode of would compare equal."""
    monkeypatch.setattr(ngspice_mode, "_flags_by_mode", {})
    for name, folder in (("spinit", "scripts"), (".spiceinit", "user")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_text("set ngbehavior=ps\n", encoding="utf-8")
    monkeypatch.setenv("SPICE_SCRIPTS", str(tmp_path / "scripts"))
    monkeypatch.setenv("SPICE_USERINIT_DIR", str(tmp_path / "user"))
    monkeypatch.chdir(tmp_path / "user")

    command = tuple(NGspiceSimulator.spice_exe)
    assert flags_for(command, mode) == flags
    assert ngspice_mode._flags_by_mode == {(command, mode): flags}


def test_an_error_asking_ngspice_leaves_the_run_unchecked_and_is_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Whatever starting the probe raises, the run is only left unchecked: a
    check that cannot run must not turn a good run into a failed one."""
    monkeypatch.setattr(ngspice_mode, "_flags_by_mode", {})
    log = tmp_path / "run.log"
    log.write_text(ORDINARY, encoding="utf-8")

    def refuse(*args, **kwargs):
        raise RuntimeError("no process may start here")

    monkeypatch.setattr(ngspice_mode.subprocess, "run", refuse)

    class Present(NGspiceSimulator):
        spice_exe: ClassVar[list[str]] = ["ngspice"]

    check = check_run(log, Present, "kiltpsa")
    assert check.status == "unchecked"
    assert check.reason
