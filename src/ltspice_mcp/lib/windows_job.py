"""Windows process ownership for workers and their descendants.

An unnamed Job Object survives its first process and retains all descendants.
Closing its non-inheritable handle terminates them, including when Windows
closes the handle because the supervisor died. Explicitly detached owners may
break away; ordinary subprocesses stay in the job.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from functools import cache

_EXTENDED_LIMIT_INFORMATION = 9
_BREAKAWAY_OK = 0x0800
_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def python_launch() -> tuple[str, dict[str, str] | None]:
    """Launch Python directly while preserving a Windows virtual environment.

    The venv redirector creates a job that blocks detached children from
    escaping. Follow multiprocessing's Windows launch path: use the base
    interpreter with the redirector's environment marker, preserving imports
    and sys.executable without inserting another supervising process.
    """
    base = getattr(sys, "_base_executable", sys.executable)
    if sys.platform == "win32" and os.path.normcase(base) != os.path.normcase(sys.executable):
        return base, {**os.environ, "__PYVENV_LAUNCHER__": sys.executable}
    return sys.executable, None


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@cache
def _kernel():
    if sys.platform != "win32":
        raise OSError("Windows Job Objects require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
            wintypes.BOOL,
        ),
        "QueryInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
            wintypes.BOOL,
        ),
        "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "IsProcessInJob": (
            [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
            wintypes.BOOL,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments
        function.restype = result
    return kernel


def _check(result):
    if not result:
        if sys.platform == "win32":
            raise ctypes.WinError(ctypes.get_last_error())
        raise OSError("Windows Job Objects require Windows")
    return result


class WindowsJob:
    """Own a process tree until close, independently of its root's lifetime."""

    def __init__(self, pid: int) -> None:
        kernel = _kernel()
        self._handle = _check(kernel.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE | _BREAKAWAY_OK
            _check(
                kernel.SetInformationJobObject(
                    self._handle,
                    _EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits),
                    ctypes.sizeof(limits),
                )
            )
            process = _check(
                kernel.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
            )
            try:
                _check(kernel.AssignProcessToJobObject(self._handle, process))
            finally:
                kernel.CloseHandle(process)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._handle is not None:
            _check(_kernel().CloseHandle(self._handle))
            self._handle = None


def detached_creation_flags() -> int:
    """Leave an enclosing worker job only when that job permits detachment."""
    if sys.platform != "win32":
        return 0
    kernel = _kernel()
    in_job = wintypes.BOOL()
    _check(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)))
    if not in_job.value:
        return 0
    limits = _ExtendedLimits()
    _check(
        kernel.QueryInformationJobObject(
            None, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits), None
        )
    )
    return (
        _CREATE_BREAKAWAY_FROM_JOB
        if limits.BasicLimitInformation.LimitFlags & _BREAKAWAY_OK
        else 0
    )
