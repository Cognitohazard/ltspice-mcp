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
from ltspice_mcp.lib.ngspice_mode import check_run, deck_flags, flags_for

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

# The head of a log ngspice-42 wrote for an ordinary run.
ORDINARY = "\nNote: Compatibility modes selected: ps lt ki a\n\n\nCircuit: * divider\n\n"
# A driver that sets hsa and sources the deck: the driver is read first.
DRIVER_ONLY = "\nNote: Compatibility modes selected: ps lt ki a\n\n\nCircuit: * driver\n\n"
DRIVER = (
    "\nNote: Compatibility modes selected: ps lt ki a\n\n\nCircuit: * driver\n\n"
    "\nNote: Compatibility modes selected: hs a\n\n\nCircuit: * m multiplier\n\n"
)


@pytest.mark.parametrize(
    ("text", "driver", "flags"),
    [
        (ORDINARY, False, ("ps", "lt", "ki", "a")),
        (ORDINARY.replace("\n", "\r\n"), False, ("ps", "lt", "ki", "a")),
        (DRIVER, True, ("hs", "a")),
        # A deck whose own .control sources another circuit: the first note is the deck's.
        (DRIVER, False, ("ps", "lt", "ki", "a")),
        (DRIVER_ONLY, True, None),
        ("\nNote: No compatibility mode selected!\n\n\nCircuit: * r\n", False, ()),
        ("\r\nNote: No compatibility mode selected!\r\n\r\nCircuit: * r\r\n", False, ()),
        ("\nCircuit: * made in mc mode\n\nDoing analysis at TEMP = 27\n", False, None),
        ("", False, None),
    ],
    ids=[
        "ordinary",
        "crlf",
        "driver-second",
        "deck-sources-another",
        "driver-deck-unread",
        "no-mode",
        "no-mode-crlf",
        "no-note",
        "empty",
    ],
)
def test_the_decks_own_note_names_its_mode(text: str, driver: bool, flags: tuple[str, ...] | None):
    assert deck_flags(text, driver=driver) == flags


def test_a_note_quoted_inside_a_line_is_not_a_note():
    text = "* title Note: Compatibility modes selected: hs a\n"
    assert deck_flags(text, driver=False) is None


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        (None, None, "no_log"),
        ("absent.log", None, "log_unreadable"),
        ("mc.log", "\nCircuit: * mc\n", "no_mode_in_log"),
        ("driver.log", DRIVER_ONLY, "deck_note_missing"),
    ],
)
def test_a_log_without_the_decks_mode_leaves_the_run_unchecked_and_says_why(
    name: str | None, content: str | None, reason: str, tmp_path: Path
):
    log = None if name is None else tmp_path / name
    if log is not None and content is not None:
        log.write_text(content, encoding="utf-8")
    check = check_run(log, NGspiceSimulator, "hsa", driver=reason == "deck_note_missing")
    assert check.status == "unchecked"
    assert check.evidence() == {"ngbehavior": "hsa", "reason": reason}
    assert check.explanation()


def test_a_simulator_that_cannot_be_asked_leaves_the_run_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ngspice_mode, "_flags_by_mode", {})
    log = tmp_path / "run.log"
    log.write_text(ORDINARY, encoding="utf-8")

    class Missing(NGspiceSimulator):
        spice_exe: ClassVar[list[str]] = [str(tmp_path / "no-such-ngspice")]

    check = check_run(log, Missing, "kiltpsa", driver=False)
    assert check.evidence() == {
        "ngbehavior": "kiltpsa",
        "reason": "ngspice_not_asked",
        "logged_modes": ["ps", "lt", "ki", "a"],
    }
    # The failure is kept, so no later run waits on it again.
    assert ngspice_mode._flags_by_mode == {((str(tmp_path / "no-such-ngspice"),), "kiltpsa"): None}


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

    check = check_run(log, Present, "kiltpsa", driver=False)
    assert check.status == "unchecked"
    assert check.reason == "ngspice_not_asked"


@needs_ngspice
@pytest.mark.parametrize("loads", [True, False], ids=["deck-loads", "deck-fails-to-load"])
async def test_a_driver_run_is_checked_on_its_decks_note(
    loads: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A native setup's driver is read in the -D mode and then sources the
    deck in the mode it sets, so the deck's note is the second. When the
    deck fails to load, ngspice prints only the driver's note, which says
    nothing about the deck's mode: the run is not blamed on the mode, and
    says why its mode went unchecked."""
    import asyncio

    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.runner_base import NativeLaunchContext, RunnerBase
    from ltspice_mcp.lib.simulator import detect_simulators
    from tests.conftest import LIVENESS_S

    monkeypatch.setattr(
        NGspiceSimulator, "_compatibility_mode", NGspiceSimulator._compatibility_mode
    )
    detect_simulators(ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path]))
    assert NGspiceSimulator._compatibility_mode == "kiltpsa"
    folder = tmp_path / "runs" / "job"
    folder.mkdir(parents=True)
    source = folder / "sample.input.cir"
    source.write_text(
        "* input\n" + ("" if loads else ".include missing.inc\n") + "V1 in 0 1\nR1 in 0 1k\n"
        ".op\n.end\n",
        encoding="utf-8",
    )
    driver = folder / "sample.setup.cir"
    driver.write_text(
        "* setup\n.control\nset ngbehavior=hsa\nsource sample.input.cir\n"
        "run\nwrite sample.raw\nquit\n.endc\n.end\n",
        encoding="utf-8",
    )
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    runner = RunnerBase(loop, NGspiceSimulator, tmp_path / "runs")
    handle = await asyncio.to_thread(
        runner.submit_netlist,
        driver,
        str(Path("job") / "sample.cir"),
        result.set_result,
        native=NativeLaunchContext(input_deck=source, cwd=folder),
    )
    outcome = await asyncio.wait_for(result, LIVENESS_S)
    for task in handle.active_tasks:
        await asyncio.to_thread(task.join, LIVENESS_S)

    codes = {item["code"] for item in outcome.observations}
    assert outcome.failure_code != "ngspice_mode_mismatch", outcome
    if loads:
        assert not outcome.error, outcome
        assert "ngspice_mode_unverified" not in codes
    else:
        (unchecked,) = [o for o in outcome.observations if o["code"] == "ngspice_mode_unverified"]
        assert unchecked["evidence"]["reason"] == "deck_note_missing"
