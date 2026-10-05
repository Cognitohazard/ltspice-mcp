"""The stat stamp that says a file has not changed since it was read.

A stamp is the file's device, inode, size, modification time and change time,
both times in nanoseconds since the Unix epoch. A write moves the modification
time, but a tool can put it back; the change time it cannot, so a rewrite that
restores the modification time still moves the stamp. POSIX reports the change
time as ``st_ctime``. Windows reports the creation time there instead, so on
Windows the change time is read directly (``FILE_BASIC_INFO.ChangeTime``), and
a filesystem that reports none there (a zero) leaves the file unstamped.

A file whose stamp cannot be read in full is not stamped at all: the caller
then has to look at its content.
"""

from __future__ import annotations

import ctypes
import os
import stat
import sys
from functools import cache

Stamp = tuple[int, int, int, int, int]

ABSENT = "absent"


def file_stamp(path: str) -> Stamp | str | None:
    """``path``'s stamp, ``ABSENT`` when there is no such file, or None when
    it cannot be stamped (its metadata cannot be read, or it is not a regular
    file)."""
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return ABSENT
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    if sys.platform == "win32":
        changed = _windows_change_time(path)
        if changed is None:
            return None
    else:
        changed = info.st_ctime_ns
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, changed


if sys.platform == "win32":
    from ctypes import wintypes

    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_ALL = 0x0001 | 0x0002 | 0x0004
    _OPEN_EXISTING = 3
    _FILE_BASIC_INFO = 0
    _UNIX_EPOCH_AS_FILETIME = 116_444_736_000_000_000
    """1970-01-01 in FILETIME's 100-nanosecond ticks since 1601-01-01."""

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    @cache
    def _kernel():
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        return kernel

    def _windows_change_time(path: str) -> int | None:
        """``path``'s change time in Unix nanoseconds, or None if the
        filesystem cannot report one.

        Opened for attributes only, sharing everything, so a simulator still
        writing the file neither blocks this nor is blocked by it.
        """
        kernel = _kernel()
        handle = kernel.CreateFileW(
            path, _FILE_READ_ATTRIBUTES, _FILE_SHARE_ALL, None, _OPEN_EXISTING, 0, None
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            info = _FileBasicInfo()
            if not kernel.GetFileInformationByHandleEx(
                handle, _FILE_BASIC_INFO, ctypes.byref(info), ctypes.sizeof(info)
            ):
                return None
            if info.ChangeTime <= 0:
                return None
            return (int(info.ChangeTime) - _UNIX_EPOCH_AS_FILETIME) * 100
        finally:
            kernel.CloseHandle(handle)
