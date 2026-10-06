"""Windows process ownership for workers and their descendants.

An unnamed Job Object survives its first process and retains all descendants.
Closing its non-inheritable handle terminates them, including when Windows
closes the handle because the supervisor died. Explicitly detached owners may
break away; ordinary subprocesses stay in the job.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from ctypes import wintypes
from functools import cache

_BASIC_ACCOUNTING_INFORMATION = 1
_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
_EXTENDED_LIMIT_INFORMATION = 9
_BREAKAWAY_OK = 0x0800
_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_MEMORY = 0x0100
_JOB_MEMORY = 0x0200
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


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


class _AssociateCompletionPort(ctypes.Structure):
    _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodUserTime", ctypes.c_longlong),
        ("ThisPeriodKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
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
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "IsProcessInJob": (
            [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
            wintypes.BOOL,
        ),
        "CreateIoCompletionPort": (
            [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD],
            wintypes.HANDLE,
        ),
        "GetQueuedCompletionStatus": (
            [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_void_p),
                wintypes.DWORD,
            ],
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
    """Own a process tree until close, independently of its root's lifetime.

    A job closed with a cleanup bound terminates its processes and waits for
    the count to reach zero. Windows posts a message to the job's completion
    port when that happens, so the wait ends with it; delivery of those
    messages is not guaranteed, so the count itself is what decides.
    """

    def __init__(
        self,
        pid: int,
        *,
        allow_breakaway: bool = True,
        memory_limit_bytes: int | None = None,
        cleanup_timeout_s: float | None = None,
    ) -> None:
        if memory_limit_bytes is not None and (
            type(memory_limit_bytes) is not int or memory_limit_bytes <= 0
        ):
            raise ValueError("Job memory limit must be a positive integer")
        if cleanup_timeout_s is not None and (
            not math.isfinite(cleanup_timeout_s) or cleanup_timeout_s <= 0
        ):
            raise ValueError("Job cleanup timeout must be finite and positive")
        self._cleanup_timeout_s = cleanup_timeout_s
        self._port = None
        kernel = _kernel()
        self._handle = _check(kernel.CreateJobObjectW(None, None))
        try:
            if cleanup_timeout_s is not None:
                self._port = _check(
                    kernel.CreateIoCompletionPort(_INVALID_HANDLE_VALUE, None, 0, 1)
                )
                association = _AssociateCompletionPort(None, self._port)
                _check(
                    kernel.SetInformationJobObject(
                        self._handle,
                        _ASSOCIATE_COMPLETION_PORT_INFORMATION,
                        ctypes.byref(association),
                        ctypes.sizeof(association),
                    )
                )
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            if allow_breakaway:
                limits.BasicLimitInformation.LimitFlags |= _BREAKAWAY_OK
            if memory_limit_bytes is not None:
                limits.BasicLimitInformation.LimitFlags |= _PROCESS_MEMORY | _JOB_MEMORY
                limits.ProcessMemoryLimit = memory_limit_bytes
                limits.JobMemoryLimit = memory_limit_bytes
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

    def active_processes(self) -> int:
        """How many processes the job holds now."""
        kernel = _kernel()
        accounting = _BasicAccounting()
        _check(
            kernel.QueryInformationJobObject(
                self._handle,
                _BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(accounting),
                ctypes.sizeof(accounting),
                None,
            )
        )
        return int(accounting.ActiveProcesses)

    def _await_empty(self, deadline: float) -> None:
        kernel = _kernel()
        message = wintypes.DWORD()
        key = ctypes.c_size_t()
        overlapped = ctypes.c_void_p()
        while self.active_processes():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Windows parser job did not become empty")
            # A timeout or an unrelated message both lead back to the count.
            kernel.GetQueuedCompletionStatus(
                self._port,
                ctypes.byref(message),
                ctypes.byref(key),
                ctypes.byref(overlapped),
                max(1, int(remaining * 1000)),
            )

    def close(self) -> None:
        if self._handle is not None:
            kernel = _kernel()
            try:
                if self._cleanup_timeout_s is not None:
                    _check(kernel.TerminateJobObject(self._handle, 1))
                    self._await_empty(time.monotonic() + self._cleanup_timeout_s)
            finally:
                _check(kernel.CloseHandle(self._handle))
                self._handle = None
                if self._port is not None:
                    kernel.CloseHandle(self._port)
                    self._port = None


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
