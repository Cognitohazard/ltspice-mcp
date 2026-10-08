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
_BASIC_PROCESS_ID_LIST = 3
_EXTENDED_LIMIT_INFORMATION = 9
_ACTIVE_PROCESS = 0x0008
_BREAKAWAY_OK = 0x0800
_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_MEMORY = 0x0100
_JOB_MEMORY = 0x0200
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_ERROR_INVALID_PARAMETER = 87
_ERROR_MORE_DATA = 234
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258


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
        "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments
        function.restype = result
    return kernel


def _last_error_code() -> int:
    return ctypes.get_last_error() if sys.platform == "win32" else 0


def _last_error() -> OSError:
    """The calling thread's last Windows error, as the exception to raise."""
    if sys.platform == "win32":
        return ctypes.WinError(ctypes.get_last_error())
    return OSError("Windows Job Objects require Windows")


def _check(result):
    if not result:
        raise _last_error()
    return result


class WindowsJob:
    """Own a process tree until close, independently of its root's lifetime.

    A job closed with a cleanup bound ends its processes and returns once each
    of them has exited. The job's own count cannot say that: it reads zero as
    soon as the job is asked to terminate, while the processes are still
    exiting with their files open. So the close seals the job against new
    processes, takes a handle to each process in it, terminates them, and
    waits on those handles.
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
        self._limits = _ExtendedLimits()
        kernel = _kernel()
        self._handle = _check(kernel.CreateJobObjectW(None, None))
        try:
            self._limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            if allow_breakaway:
                self._limits.BasicLimitInformation.LimitFlags |= _BREAKAWAY_OK
            if memory_limit_bytes is not None:
                self._limits.BasicLimitInformation.LimitFlags |= _PROCESS_MEMORY | _JOB_MEMORY
                self._limits.ProcessMemoryLimit = memory_limit_bytes
                self._limits.JobMemoryLimit = memory_limit_bytes
            self._set_limits()
            process = _check(
                kernel.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
            )
            try:
                _check(kernel.AssignProcessToJobObject(self._handle, process))
            finally:
                kernel.CloseHandle(process)
        except BaseException:
            # The job holds no process, so there is none to end or wait for.
            self._release()
            raise

    def _set_limits(self) -> None:
        _check(
            _kernel().SetInformationJobObject(
                self._handle,
                _EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(self._limits),
                ctypes.sizeof(self._limits),
            )
        )

    def active_processes(self) -> int:
        """How many processes the job counts as active.

        A process that exits, or is ended by itself, leaves the count as it
        exits. A job asked to terminate counts none from that moment, while
        its processes are still exiting.
        """
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

    def seal(self) -> None:
        """Admit no further process: a start by one already in the job fails.

        The processes in the job are left as they are.
        """
        self._limits.BasicLimitInformation.LimitFlags |= _ACTIVE_PROCESS
        self._limits.BasicLimitInformation.ActiveProcessLimit = 1
        self._set_limits()

    def _process_ids(self) -> list[int]:
        kernel = _kernel()
        room = 1  # A tree is usually one process; the job says when there are more.
        while True:

            class ProcessIdList(ctypes.Structure):
                _fields_ = [
                    ("NumberOfAssignedProcesses", wintypes.DWORD),
                    ("NumberOfProcessIdsInList", wintypes.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * room),
                ]

            listed = ProcessIdList()
            if kernel.QueryInformationJobObject(
                self._handle,
                _BASIC_PROCESS_ID_LIST,
                ctypes.byref(listed),
                ctypes.sizeof(listed),
                None,
            ):
                return listed.ProcessIdList[: listed.NumberOfProcessIdsInList]
            if _last_error_code() != _ERROR_MORE_DATA:
                raise _last_error()
            room = max(listed.NumberOfAssignedProcesses, room + 1)

    def _open_members(self, handles: list[int]) -> None:
        """Add to ``handles`` a handle to each process in the job."""
        kernel = _kernel()
        for pid in self._process_ids():
            handle = kernel.OpenProcess(
                _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                if _last_error_code() == _ERROR_INVALID_PARAMETER:
                    continue  # It exited after the list was read.
                raise _last_error()
            handles.append(handle)
            member = wintypes.BOOL()
            _check(kernel.IsProcessInJob(handle, self._handle, ctypes.byref(member)))
            if not member.value:
                # It exited too, and its id names another process by now.
                kernel.CloseHandle(handles.pop())

    def _await_exit(self, members: list[int], deadline: float) -> None:
        kernel = _kernel()
        for member in members:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            result = kernel.WaitForSingleObject(member, remaining_ms)
            if result == _WAIT_TIMEOUT:
                raise TimeoutError("A process in the Windows job did not exit")
            if result != _WAIT_OBJECT_0:
                raise _last_error()

    def _release(self) -> None:
        """Close the job's handle, which ends any process still in it."""
        if self._handle is not None:
            _check(_kernel().CloseHandle(self._handle))
            self._handle = None

    def close(self) -> None:
        if self._handle is None:
            return
        kernel = _kernel()
        members: list[int] = []
        try:
            if self._cleanup_timeout_s is not None:
                deadline = time.monotonic() + self._cleanup_timeout_s
                self.seal()  # The processes listed next are all there will be.
                self._open_members(members)
                _check(kernel.TerminateJobObject(self._handle, 1))
                self._await_exit(members, deadline)
        finally:
            for member in members:
                kernel.CloseHandle(member)
            self._release()


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
