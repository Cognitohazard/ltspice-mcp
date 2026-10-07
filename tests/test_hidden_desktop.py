"""Starting a program on a desktop of its own (``lib/hidden_desktop.py``).

A real process is started each time, and says for itself which desktop it is
on: nothing here stands in for ``CreateProcessW``. Windows only; elsewhere the
module hands out no desktop, which the last test pins.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from ltspice_mcp.lib import hidden_desktop
from ltspice_mcp.lib.hidden_desktop import BoxWatch, DialogError, HiddenDesktop
from ltspice_mcp.lib.windows_job import python_launch
from tests.conftest import identify, process_running, wait_until, written


def _no_desktop_here() -> str | None:
    """Why no desktop can be made here, or None when one can."""
    if sys.platform != "win32":
        return "a desktop of one's own is a Windows facility"
    with HiddenDesktop(f"ltspice-mcp-test-probe-{os.getpid()}") as made:
        if not made.available:
            return "Windows refuses this session a desktop of its own (it is not interactive)"
    return None


# Where the launch falls back to the ordinary one, there is nothing to test.
_NO_DESKTOP = _no_desktop_here()
windows_only = pytest.mark.skipif(_NO_DESKTOP is not None, reason=_NO_DESKTOP or "")


def own_desktop() -> str:
    """The name of the desktop the calling thread is on."""
    assert sys.platform == "win32"
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32")
    kernel = ctypes.WinDLL("kernel32")
    user.GetThreadDesktop.restype = wintypes.HANDLE
    user.GetThreadDesktop.argtypes = [wintypes.DWORD]
    user.GetUserObjectInformationW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    name = ctypes.create_unicode_buffer(256)
    desktop = user.GetThreadDesktop(kernel.GetCurrentThreadId())
    user.GetUserObjectInformationW(desktop, 2, name, ctypes.sizeof(name), None)
    return name.value


#: The start of a program that has to say which desktop it runs on: the
#: function above, as the test itself calls it.
OWN_DESKTOP_SOURCE = "import sys\n\n\n" + inspect.getsource(own_desktop)

# A program that says where it is running, then does what its first argument
# names: report and exit 7, wait to be ended, or put up a message box.
PROBE = OWN_DESKTOP_SOURCE + textwrap.dedent(
    """
    import ctypes, json, os, threading

    mode, report = sys.argv[1], sys.argv[2]
    facts = {
        "desktop": own_desktop(),
        "cwd": os.getcwd(),
        "probe": os.environ.get("HIDDEN_DESKTOP_PROBE"),
        "pid": os.getpid(),
    }
    with open(report, "w", encoding="utf-8") as out:
        json.dump(facts, out)
    if mode == "report":
        print("to stdout")
        print("to stderr", file=sys.stderr)
        sys.exit(7)
    if mode == "ask":
        ctypes.WinDLL("user32").MessageBoxW(None, "Aborting: Unknown schematic syntax", "probe", 0)
    threading.Event().wait()
    """
)


@pytest.fixture
def desktop() -> Iterator[HiddenDesktop]:
    with HiddenDesktop(f"ltspice-mcp-test-{os.getpid()}") as made:
        yield made


@pytest.fixture
def probe(tmp_path: Path):
    """The command line of the probe for a mode, and where it reports."""
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    report = tmp_path / "report.json"
    python, _ = python_launch()

    def command(mode: str) -> list[str]:
        return [python, str(script), mode, str(report)]

    command.report = report  # type: ignore[attr-defined]
    return command


def environment(**extra: str) -> dict[str, str]:
    """This process's environment as the base interpreter needs it, plus ``extra``."""
    _, launch = python_launch()
    return {**(launch or os.environ), **extra}


@windows_only
class TestStart:
    def test_a_program_runs_on_the_hidden_desktop_and_not_on_the_callers(
        self, desktop: HiddenDesktop, probe, tmp_path: Path
    ):
        code = desktop.run(probe("report"), cwd=tmp_path, env=environment())
        facts = json.loads(probe.report.read_text(encoding="utf-8"))
        assert code == 7
        assert facts["desktop"] == desktop.name
        assert facts["desktop"] != own_desktop()

    def test_the_working_directory_and_environment_are_the_ones_given(
        self, desktop: HiddenDesktop, probe, tmp_path: Path
    ):
        folder = tmp_path / "模型 Zoë"
        folder.mkdir()
        desktop.run(probe("report"), cwd=folder, env=environment(HIDDEN_DESKTOP_PROBE="值 ë"))
        facts = json.loads(probe.report.read_text(encoding="utf-8"))
        assert Path(facts["cwd"]) == folder
        assert facts["probe"] == "值 ë"

    def test_the_streams_named_are_the_ones_written(
        self, desktop: HiddenDesktop, probe, tmp_path: Path
    ):
        both = tmp_path / "both.log"
        with open(both, "wb") as console:
            desktop.run(
                probe("report"), env=environment(), stdout=console, stderr=subprocess.STDOUT
            )
        # Either order: Python flushes its buffered stdout at exit, after stderr.
        assert sorted(both.read_bytes().splitlines()) == [b"to stderr", b"to stdout"]
        out, err = tmp_path / "out.log", tmp_path / "err.log"
        with open(out, "wb") as to_out, open(err, "wb") as to_err:
            desktop.run(probe("report"), env=environment(), stdout=to_out, stderr=to_err)
        assert out.read_bytes().strip() == b"to stdout"
        assert err.read_bytes().strip() == b"to stderr"

    def test_a_handle_not_named_is_not_inherited(
        self, desktop: HiddenDesktop, probe, tmp_path: Path
    ):
        """A file this process holds open, and has even marked inheritable,
        stays out of the program: only the streams it was given go in. Windows
        refuses to delete a file any process still holds open."""
        assert sys.platform == "win32"
        import msvcrt

        held = tmp_path / "held.txt"
        handle = open(held, "wb")  # noqa: SIM115 - held open across the launch on purpose
        os.set_handle_inheritable(msvcrt.get_osfhandle(handle.fileno()), True)
        with open(tmp_path / "console.log", "wb") as console:
            started = desktop.start(probe("wait"), env=environment(), stdout=console)
        try:
            wait_until(written(probe.report, json.loads), what="the probe to report")
            handle.close()
            held.unlink()
        finally:
            started.close()

    def test_past_its_timeout_the_program_is_ended(self, desktop: HiddenDesktop, probe):
        with pytest.raises(subprocess.TimeoutExpired):
            # timing: the bound is the behaviour under test; the probe never exits
            desktop.run(probe("wait"), timeout=1.0, env=environment())
        pid = json.loads(probe.report.read_text(encoding="utf-8"))["pid"]
        wait_until(lambda: not process_running(pid), what="the probe to be gone")

    def test_closing_a_started_program_ends_it(self, desktop: HiddenDesktop, probe):
        started = desktop.start(probe("wait"), env=environment())
        facts = wait_until(written(probe.report, json.loads), what="the probe to report")
        running = identify(facts["pid"])
        assert running is not None
        assert started.poll() is None
        started.close()
        wait_until(lambda: not running.is_running(), what="the probe to be gone")

    @pytest.mark.usefixtures("quick_looks")
    def test_a_program_waiting_on_a_message_box_is_ended_with_what_it_said(
        self, desktop: HiddenDesktop, probe
    ):
        command = probe("ask")
        with pytest.raises(DialogError) as stopped:
            desktop.run(command, env=environment(), remedy="Look for yourself")
        assert stopped.value.text == "probe\nAborting: Unknown schematic syntax"
        assert str(stopped.value).startswith(f"{Path(command[0]).name} stopped on a message box")
        assert str(stopped.value).endswith(
            "probe; Aborting: Unknown schematic syntax. Look for yourself"
        )
        pid = json.loads(probe.report.read_text(encoding="utf-8"))["pid"]
        wait_until(lambda: not process_running(pid), what="the probe to be gone")

    def test_a_program_that_cannot_be_started_raises(self, desktop: HiddenDesktop, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            desktop.start([str(tmp_path / "no-such-program.exe")])

    def test_a_closed_desktop_starts_nothing(self, desktop: HiddenDesktop, probe):
        desktop.close()
        assert not desktop.available
        with pytest.raises(OSError, match="No desktop to start a program on"):
            desktop.start(probe("report"))


class TestBoxWatch:
    """A box counts once the same one has been there on two looks running."""

    class Desktop:
        """What a desktop says a program's message box reads, one look at a time."""

        def __init__(self, *looks: str | None) -> None:
            self._looks = iter(looks)

        def dialog(self, _pid: int, _ignore: str | None = None) -> str | None:
            return next(self._looks)

    def watch(self, *looks: str | None) -> list[str | None]:
        boxes = BoxWatch(self.Desktop(*looks), 1)  # type: ignore[arg-type]
        return [boxes.look() for _ in looks]

    def test_a_box_seen_once_is_not_yet_a_question(self):
        assert self.watch(None, "Save?") == [None, None]

    def test_the_same_box_on_the_next_look_is(self):
        assert self.watch("Save?", "Save?") == [None, "Save?"]

    def test_a_box_that_closed_by_itself_never_counts(self):
        assert self.watch("Loading", None, "Loading", None) == [None, None, None, None]

    def test_another_box_starts_the_count_again(self):
        assert self.watch("Save?", "Sure?", "Sure?") == [None, None, "Sure?"]


@windows_only
class TestSharedDesktop:
    def test_one_desktop_is_made_on_first_use_and_kept(self):
        first = hidden_desktop.shared()
        assert first is not None
        assert first.available
        assert first.name == f"ltspice-mcp-{os.getpid()}"
        assert hidden_desktop.shared() is first

    def test_closing_it_lets_the_next_use_make_another(self):
        first = hidden_desktop.shared()
        assert first is not None
        hidden_desktop.close_shared()
        assert not first.available
        second = hidden_desktop.shared()
        assert second is not None
        assert second is not first

    def test_turned_off_there_is_none(self):
        hidden_desktop.configure(enabled=False)
        assert hidden_desktop.shared() is None
        hidden_desktop.configure(enabled=True)
        assert hidden_desktop.shared() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="what the module does where there is none")
def test_off_windows_there_is_no_desktop_to_start_on():
    assert hidden_desktop.shared() is None
    desktop = HiddenDesktop("ltspice-mcp-test")
    assert not desktop.available
    assert desktop.dialog(os.getpid()) is None
    assert desktop.window_owners() == set()
    with pytest.raises(OSError, match="No desktop to start a program on"):
        desktop.run([sys.executable, "-c", "pass"])
