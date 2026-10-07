"""Starting a Windows program where it cannot take the keyboard focus.

LTspice opens a window even for a batch run or an export and is the foreground
window for most of it, whatever show state it is started with: started
minimised (``SW_SHOWMINNOACTIVE``) or hidden (``SW_HIDE``) it still held the
foreground for about four samples in five of a run. A person typing in another
program loses the keyboard on every simulation, and a sweep takes it
continuously.

A process started on another desktop has its windows there and nowhere else,
so it cannot take the focus of the desktop someone is working at: started so,
neither LTspice 26 nor LTspice XVII was ever the foreground window, and both
ran, exported and wrote their files as usual. ``subprocess`` cannot ask for
that, because ``STARTUPINFO.lpDesktop`` is not among the fields it passes on,
so this module calls ``CreateProcessW`` itself.

Nobody can answer a message box there either. LTspice puts one up for some
inputs and waits for OK; ``HiddenDesktop.run`` looks for one while it waits,
and ends a program found waiting on the same box twice, with what the box
said. Every process started here is held in a job that ends with its handle,
so a program no one can see cannot outlive the process that started it.

Off Windows nothing here does anything: ``shared`` returns None and a caller
launches the ordinary way.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from ctypes import wintypes
from functools import cache
from pathlib import Path
from typing import IO, Any, Self

from ltspice_mcp.lib.windows_job import WindowsJob

logger = logging.getLogger(__name__)

_GENERIC_ALL = 0x10000000
_WAIT_OBJECT_0 = 0
_INFINITE = 0xFFFFFFFF
# The longest wait a 32-bit count of milliseconds can state short of "forever".
_LONGEST_WAIT_MS = 0xFFFFFFFE
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_STARTF_USESTDHANDLES = 0x00000100
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_DUPLICATE_SAME_ACCESS = 0x00000002
# The window class of a dialog box.
_DIALOG_CLASS = "#32770"

DIALOG_LOOK_S = 0.5
"""How often ``HiddenDesktop.run`` looks for a message box while the program
runs. A box has to be there on two looks running to count (``BoxWatch``), so
one that closes by itself within this long is never taken for a question."""


class DialogError(RuntimeError):
    """A program was ended because it was waiting on a message box.

    ``text`` is the box's title and then each line in it. ``remedy`` is how
    the caller's user gets to see such a box, for the message.
    """

    def __init__(self, program: str, text: str, remedy: str = "") -> None:
        self.text = text
        shown = "; ".join(line.strip() for line in text.splitlines() if line.strip())
        super().__init__(
            f"{program} stopped on a message box and was ended, because it runs where "
            f"no one can answer one. The box said: {shown}" + (f". {remedy}" if remedy else "")
        )


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


_NEEDS_WINDOWS = "A desktop of its own for a program needs Windows"
_WM_COMMAND = 0x0111


@cache
def _kernel() -> Any:
    if sys.platform != "win32":
        raise OSError(_NEEDS_WINDOWS)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures: dict[str, tuple[list[Any], Any]] = {
        "CreateProcessW": (
            [
                wintypes.LPCWSTR,
                wintypes.LPWSTR,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.BOOL,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessInformation),
            ],
            wintypes.BOOL,
        ),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
        "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        "GetExitCodeProcess": (
            [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
            wintypes.BOOL,
        ),
        "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "DuplicateHandle": (
            [
                wintypes.HANDLE,
                wintypes.HANDLE,
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.HANDLE),
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ],
            wintypes.BOOL,
        ),
        "InitializeProcThreadAttributeList": (
            [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)],
            wintypes.BOOL,
        ),
        "UpdateProcThreadAttribute": (
            [
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
            wintypes.BOOL,
        ),
        "DeleteProcThreadAttributeList": ([ctypes.c_void_p], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments
        function.restype = result
    return kernel


@cache
def _window_callback() -> Any:
    """The type of the function Windows calls with each window it lists."""
    if sys.platform != "win32":
        raise OSError(_NEEDS_WINDOWS)
    return ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


@cache
def _user() -> Any:
    if sys.platform != "win32":
        raise OSError(_NEEDS_WINDOWS)
    user = ctypes.WinDLL("user32", use_last_error=True)
    window_callback = _window_callback()
    signatures: dict[str, tuple[list[Any], Any]] = {
        "CreateDesktopW": (
            [
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
            ],
            wintypes.HANDLE,
        ),
        "CloseDesktop": ([wintypes.HANDLE], wintypes.BOOL),
        "EnumWindows": ([window_callback, wintypes.LPARAM], wintypes.BOOL),
        "EnumDesktopWindows": ([wintypes.HANDLE, window_callback, wintypes.LPARAM], wintypes.BOOL),
        "EnumChildWindows": ([wintypes.HWND, window_callback, wintypes.LPARAM], wintypes.BOOL),
        "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "GetClassNameW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "GetWindowThreadProcessId": (
            [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
            wintypes.DWORD,
        ),
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "PostMessageW": (
            [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM],
            wintypes.BOOL,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(user, name)
        function.argtypes = arguments
        function.restype = result
    return user


def _listed(enumerate_with: Callable[[Any], object]) -> list[tuple[int, int]]:
    """Each window an ``Enum...Windows`` call lists, with the process that owns it."""
    user = _user()
    windows: list[tuple[int, int]] = []

    def note(window: int, _unused: int) -> bool:
        owner = wintypes.DWORD(0)
        user.GetWindowThreadProcessId(window, ctypes.byref(owner))
        windows.append((window, int(owner.value)))
        return True

    enumerate_with(_window_callback()(note))
    return windows


def window_text(window: int) -> str:
    """A window's text: a top-level window's title, a control's label or contents."""
    buffer = ctypes.create_unicode_buffer(2048)
    _user().GetWindowTextW(window, buffer, len(buffer))
    return buffer.value


def window_class(window: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    _user().GetClassNameW(window, buffer, len(buffer))
    return buffer.value


def child_windows(window: int) -> list[int]:
    """Every window inside ``window``, at any depth."""
    return [
        child for child, _owner in _listed(lambda note: _user().EnumChildWindows(window, note, 0))
    ]


def windows_here(pid: int) -> list[int]:
    """The top-level windows ``pid`` has on the desktop this process runs on."""
    return [
        window
        for window, owner in _listed(lambda note: _user().EnumWindows(note, 0))
        if owner == pid
    ]


def post_command(window: int, command: int) -> None:
    """Hand ``window`` a command of its own menu, as choosing the item would.

    Posted, not sent: the window carries it out in its own time, and a window
    that has stopped answering does not hold the caller. It works across
    desktops, which a window message does not always. Raises ``OSError`` when
    Windows refuses the message.
    """
    if not _user().PostMessageW(window, _WM_COMMAND, command, 0):
        raise _last_error()


def _last_error() -> OSError:
    if sys.platform != "win32":
        return OSError(_NEEDS_WINDOWS)
    return ctypes.WinError(ctypes.get_last_error())


class StartedProcess:
    """A process started on a desktop of its own: the part of ``Popen`` a caller uses.

    It is held in a job that ends when this is closed, so neither it nor
    anything it started is left running where no one can see it.
    """

    def __init__(self, handle: int, pid: int, command: Sequence[str]) -> None:
        self._handle: int | None = handle
        self.pid = pid
        self.args = list(command)
        self.returncode: int | None = None
        self._job: WindowsJob | None = None
        # Without a job the process is still ended by the caller's own waits
        # and kills; only an abandoned one could then outlive this process.
        with contextlib.suppress(OSError):
            self._job = WindowsJob(pid, allow_breakaway=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _settle(self, milliseconds: int) -> int | None:
        if self.returncode is not None or self._handle is None:
            return self.returncode
        kernel = _kernel()
        if kernel.WaitForSingleObject(self._handle, milliseconds) != _WAIT_OBJECT_0:
            return None
        code = wintypes.DWORD(0)
        kernel.GetExitCodeProcess(self._handle, ctypes.byref(code))
        self.returncode = code.value
        return self.returncode

    def poll(self) -> int | None:
        return self._settle(0)

    def wait(self, timeout: float | None = None) -> int:
        """The exit code, raising ``subprocess.TimeoutExpired`` when it does not come in time."""
        if timeout is None:
            milliseconds = _INFINITE
        else:
            milliseconds = min(_LONGEST_WAIT_MS, max(0, int(timeout * 1000)))
        code = self._settle(milliseconds)
        if code is None:
            raise subprocess.TimeoutExpired(self.args, timeout or 0.0)
        return code

    def kill(self) -> None:
        if self._handle is not None and self.returncode is None:
            _kernel().TerminateProcess(self._handle, 1)

    def close(self) -> None:
        """Release the process, ending it and what it started if still running."""
        if self._job is not None:
            with contextlib.suppress(OSError):
                self._job.close()
            self._job = None
        if self._handle is not None:
            _kernel().CloseHandle(self._handle)
        self._handle = None


def _environment_block(env: Mapping[str, str]) -> Any:
    return ctypes.create_unicode_buffer("".join(f"{key}={value}\0" for key, value in env.items()))


@contextlib.contextmanager
def _inheritable(stream: IO[Any] | int) -> Iterator[int]:
    """A copy of ``stream``'s handle that a new process can inherit, closed on exit."""
    if sys.platform != "win32":
        raise OSError(_NEEDS_WINDOWS)
    import msvcrt

    kernel = _kernel()
    current = kernel.GetCurrentProcess()
    descriptor = stream if isinstance(stream, int) else stream.fileno()
    duplicate = wintypes.HANDLE()
    if not kernel.DuplicateHandle(
        current,
        msvcrt.get_osfhandle(descriptor),
        current,
        ctypes.byref(duplicate),
        0,
        True,
        _DUPLICATE_SAME_ACCESS,
    ):
        raise _last_error()
    assert duplicate.value is not None
    try:
        yield duplicate.value
    finally:
        kernel.CloseHandle(duplicate.value)


@contextlib.contextmanager
def _handle_list(handles: Sequence[int]) -> Iterator[int]:
    """The address of an attribute list naming the only handles a new process inherits."""
    kernel = _kernel()
    size = ctypes.c_size_t(0)
    kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    attributes = ctypes.create_string_buffer(size.value)
    if not kernel.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)):
        raise _last_error()
    # Windows keeps a pointer to this array until the list is deleted.
    named = (wintypes.HANDLE * len(handles))(*handles)
    try:
        if not kernel.UpdateProcThreadAttribute(
            attributes,
            0,
            _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            named,
            ctypes.sizeof(named),
            None,
            None,
        ):
            raise _last_error()
        yield ctypes.addressof(attributes)
    finally:
        kernel.DeleteProcThreadAttributeList(attributes)


class HiddenDesktop:
    """A desktop of its own, for programs whose windows nobody should see.

    ``available`` is False off Windows and where Windows refuses a desktop (a
    session that is not interactive, or one out of desktop heap); ``start``
    raises there.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._handle: int | None = None
        if sys.platform != "win32":
            return
        self._handle = _user().CreateDesktopW(name, None, None, 0, _GENERIC_ALL, None) or None
        if self._handle is None:
            logger.warning("Windows refused a desktop named %s: %s", name, _last_error())

    @property
    def available(self) -> bool:
        return self._handle is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            _user().CloseDesktop(self._handle)
        self._handle = None

    def _top_level_windows(self) -> list[tuple[int, int]]:
        """Each top-level window here, with the process that owns it."""
        if self._handle is None:
            return []
        handle = self._handle
        return _listed(lambda note: _user().EnumDesktopWindows(handle, note, 0))

    def window_owners(self) -> set[int]:
        """The process of every top-level window on this desktop."""
        return {owner for _window, owner in self._top_level_windows()}

    def windows(self, pid: int) -> list[int]:
        """The top-level windows ``pid`` has where it was started.

        That is this desktop while it is available. When it is not, a program
        is started the ordinary way, and its windows are on the desktop this
        process runs on.
        """
        if self._handle is not None:
            listed = self._top_level_windows()
        elif sys.platform == "win32":
            listed = _listed(lambda note: _user().EnumWindows(note, 0))
        else:
            return []
        return [window for window, owner in listed if owner == pid]

    def dialog(self, pid: int, ignore: str | None = None) -> str | None:
        """What a message box ``pid`` has open says, or None when it has none.

        The box's title, then each line of text in it. Its buttons are left
        out: what was asked matters, not how it could be answered. A dialog
        titled ``ignore`` is one the caller opened itself, not a box.
        """
        found: list[str] = []
        for window in self.windows(pid):
            title = window_text(window)
            if (
                window_class(window) == _DIALOG_CLASS
                and _user().IsWindowVisible(window)
                and title != ignore
            ):
                found.append(title)
                for child in child_windows(window):
                    text = window_text(child)
                    if window_class(child) == "Static" and text.strip():
                        found.append(text)
        return "\n".join(found) if found else None

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stdin: IO[Any] | int | None = None,
        stdout: IO[Any] | int | None = None,
        stderr: IO[Any] | int | None = None,
    ) -> StartedProcess:
        """Start ``command`` with its windows on this desktop.

        ``stdin`` is an open file (or descriptor) the program reads, and
        ``stdout`` and ``stderr`` ones it writes to, ``stderr`` also
        ``subprocess.STDOUT`` for the same one as ``stdout``; None leaves the
        stream unconnected. Only those handles are inherited, never another
        this process happens to hold open, so the far end of a pipe given
        here stays this process's alone. The program is in its job before it
        runs its first instruction, and whatever it starts shares both its
        job and this desktop.
        """
        if self._handle is None:
            raise OSError("No desktop to start a program on")
        kernel = _kernel()
        startup = _StartupInfoEx()
        startup.StartupInfo.lpDesktop = self.name
        startup.StartupInfo.cb = ctypes.sizeof(_StartupInfo)
        flags = _CREATE_SUSPENDED
        block = None
        if env is not None:
            block = _environment_block(env)
            flags |= _CREATE_UNICODE_ENVIRONMENT
        created = _ProcessInformation()
        line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
        with contextlib.ExitStack() as held:
            read = None if stdin is None else held.enter_context(_inheritable(stdin))
            out = None if stdout is None else held.enter_context(_inheritable(stdout))
            if stderr == subprocess.STDOUT:
                error = out
            else:
                error = None if stderr is None else held.enter_context(_inheritable(stderr))
            inherited = [
                handle for handle in dict.fromkeys((read, out, error)) if handle is not None
            ]
            if inherited:
                # The extended form of the startup information, which is what
                # carries the list of the handles to inherit.
                startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
                startup.StartupInfo.hStdInput = read
                startup.StartupInfo.hStdOutput = out
                startup.StartupInfo.hStdError = error
                startup.lpAttributeList = held.enter_context(_handle_list(inherited))
                flags |= _EXTENDED_STARTUPINFO_PRESENT
            if not kernel.CreateProcessW(
                None,
                line,
                None,
                None,
                bool(inherited),
                flags,
                block,
                None if cwd is None else os.fspath(cwd),
                ctypes.byref(startup),
                ctypes.byref(created),
            ):
                raise _last_error()
        try:
            process = StartedProcess(created.hProcess, created.dwProcessId, command)
        except BaseException:
            # Never leave it suspended, where it would wait for ever unseen.
            kernel.TerminateProcess(created.hProcess, 1)
            kernel.CloseHandle(created.hProcess)
            raise
        finally:
            kernel.ResumeThread(created.hThread)
            kernel.CloseHandle(created.hThread)
        return process

    def run(
        self,
        command: Sequence[str],
        *,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stdout: IO[Any] | int | None = None,
        stderr: IO[Any] | int | None = None,
        remedy: str = "",
    ) -> int:
        """Run ``command`` here to its end and return its exit code.

        The contract of ``subprocess.run(...).returncode``: past ``timeout``
        the program is ended and ``subprocess.TimeoutExpired`` raised. A
        program found waiting on a message box is ended and ``DialogError``
        raised with what the box said, since no one could have answered it;
        ``remedy`` ends that error's message.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.start(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr) as process:
            boxes = BoxWatch(self, process.pid)
            while True:
                look = DIALOG_LOOK_S
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        raise subprocess.TimeoutExpired(list(command), timeout or 0.0)
                    look = min(look, remaining)
                try:
                    return process.wait(look)
                except subprocess.TimeoutExpired:
                    pass
                waiting_on = boxes.look()
                if waiting_on is not None:
                    process.kill()
                    process.wait()
                    raise DialogError(Path(command[0]).name, waiting_on, remedy)


class BoxWatch:
    """Tells a message box a program waits on from one that closes by itself.

    ``look`` gives what the box says once the same box has been there on two
    looks running, and None until then. A dialog titled ``ignore`` is one the
    caller opens itself and never counts.
    """

    def __init__(self, desktop: HiddenDesktop, pid: int, ignore: str | None = None) -> None:
        self._desktop = desktop
        self._pid = pid
        self._ignore = ignore
        self._last: str | None = None

    def look(self) -> str | None:
        seen = self._desktop.dialog(self._pid, self._ignore)
        waited_on = seen if seen is not None and seen == self._last else None
        self._last = seen
        return waited_on


# --------------------------------------------------------------------------
# The one desktop a process starts its programs on
# --------------------------------------------------------------------------

_lock = threading.Lock()
# Kept even when Windows refused it: asking again on every launch would be
# refused, and logged, every time.
_shared: HiddenDesktop | None = None
_enabled = True


def configure(*, enabled: bool) -> None:
    """Whether ``shared`` hands out a desktop at all (the ``hidden_desktop`` setting)."""
    global _enabled
    with _lock:
        _enabled = enabled


def shared() -> HiddenDesktop | None:
    """This process's hidden desktop, made on first use, or None where there is none.

    None off Windows, when the setting turns it off, and when Windows refuses
    one; a caller then launches the ordinary way.
    """
    global _shared
    if sys.platform != "win32":
        return None
    with _lock:
        if not _enabled:
            return None
        if _shared is None:
            _shared = HiddenDesktop(f"ltspice-mcp-{os.getpid()}")
        return _shared if _shared.available else None


def close_shared() -> None:
    """Close this process's hidden desktop; the next ``shared`` makes another."""
    global _shared
    with _lock:
        desktop, _shared = _shared, None
    if desktop is not None:
        desktop.close()
