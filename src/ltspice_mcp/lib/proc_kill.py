"""Scoped simulator-process termination by job-id token.

spicelib's ``kill_all_spice`` is name-global — it would terminate every
simulator process on the machine, including ones launched by a *different*
MCP server running in parallel in the same directory. (With the installed
spicelib it also matches nothing at all: it reads the base ``Simulator``
class's empty ``process_name``, so cancel/timeout silently failed to kill
the simulator everywhere the WSL taskkill path didn't apply.)

This module replaces it with a match that can only hit a job's own
process(es): every staged run netlist embeds the uuid-unique job id in its
filename (``run_filename=f"{job_id}{ext}"`` — see sim_runner and
``batch_run_filename``), so the id appears in the simulator's command line.
A process dies only when its command line carries the token AND it is
recognizably the simulator's executable.

The WSL + LTspice case is different — there the simulator is a *Windows*
process, invisible to the Linux psutil table; ``wsl.kill_windows_ltspice_by_token``
covers it with the same token idea via PowerShell + taskkill. Both kills are
attempted by the runners; on any given platform at most one finds a match.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
from collections.abc import Collection
from enum import Enum
from pathlib import Path, PurePath

import psutil

logger = logging.getLogger(__name__)


def simulator_executable_names(simulator_class: type) -> frozenset[str]:
    """Candidate executable basenames for a spicelib simulator class.

    Draws from ``spice_exe`` (the launch argv — under Wine that is
    ``["wine", ".../LTspice.exe"]``, so both basenames are included) and
    ``process_name`` when the class sets one. Lower-cased for matching.
    """
    names: set[str] = set()
    for part in getattr(simulator_class, "spice_exe", None) or []:
        name = PurePath(str(part)).name.lower()
        if name:
            names.add(name)
    process_name = str(getattr(simulator_class, "process_name", "") or "")
    if process_name:
        names.add(process_name.lower())
    return frozenset(names)


def token_in_argument(token: str, arg: str) -> bool:
    """True when ``token`` appears at a run-filename boundary in ``arg``.

    Staged run files are ``{job_id}.{ext}`` (single runs) or
    ``{job_id}_{n}.{ext}`` (batch sub-runs), so the id is always followed by
    ``.`` or ``_`` — or ends the argument.

    That trailing anchor is what makes the match safe, and nothing else here
    substitutes for it: without it a token would also match every longer id it
    happens to prefix (``{id}_case_1`` against ``{id}_case_10``), killing a
    sibling job's simulator. Do not drop it. In particular, ids are NOT all the
    same shape — ``sweep_utils.generate_id`` emits ``{prefix}_{stem}_{ts}_{hex}``
    and, when no stem survives sanitization, ``{prefix}_{ts}_{hex}`` — so no
    safety argument is available from "every id has its separators in the same
    places". What the id format does contribute is a supporting invariant:
    ``sanitize_stem`` strips ``_`` out of the stem, so an id can never grow an
    extra ``_``-delimited field, which is what would let one whole id extend
    another at exactly this boundary.
    """
    return re.search(re.escape(token) + r"(?:[._]|$)", arg) is not None


def _names_run_deck(token: str, arg: str) -> bool:
    """True when ``arg`` is the path of ``token``'s own run deck.

    LTspice runs only ``.cir``/``.net``/``.sp`` decks, and every case's deck is
    ``{token}`` plus one of them, so a process launched for the case carries
    that path while one that merely mentions the token does not. The server's
    WSL process query carries ``*{token}.*``, and a script named after a case
    carries ``{token}.py``.
    """
    return re.search(re.escape(token) + r"\.(?:cir|net|sp)$", arg, re.IGNORECASE) is not None


def _own_descendant_pids() -> frozenset[int]:
    """Every process this server started, directly or through another."""
    try:
        return frozenset(child.pid for child in psutil.Process().children(recursive=True))
    except psutil.Error:
        return frozenset()


def kill_simulator_by_token(token: str, executable_names: Collection[str]) -> int:
    """Kill local simulator processes whose command line carries ``token``.

    A process is killed only if BOTH hold:

    - some command-line argument contains ``token`` (the uuid-unique job id,
      present because the staged netlist filename embeds it) at a filename
      boundary (see ``token_in_argument``), and
    - the process is the simulator: its name — or the basename of one of its
      first two argv entries (the Wine case: argv is ``wine …/LTspice.exe``)
      — is in ``executable_names``; or it descends from this server process
      and names the token's run deck (``_names_run_deck``).

    The name gate is what keeps this safe against incidental token matches
    (e.g. the server's own WSL PowerShell interop helpers carry the token in
    their command line but are never named like a simulator). The second route
    is for a simulator this server launched under a name the gate does not
    know: a configured wrapper or launcher script, and the simulator it starts.
    It cannot reach another session's processes, which are not this server's
    descendants, and the deck-path test keeps it off the interop helpers.

    Best-effort: per-process psutil errors are skipped. Returns the number
    of processes killed.
    """
    wanted = {n.lower() for n in executable_names if n}
    if not token or not wanted:
        return 0
    own = _own_descendant_pids()
    killed = 0
    for proc in psutil.process_iter(("name", "cmdline")):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not any(token_in_argument(token, arg) for arg in cmdline):
                continue
            candidates = {(proc.info.get("name") or "").lower()}
            candidates.update(PurePath(arg).name.lower() for arg in cmdline[:2])
            launched_here = proc.pid in own and any(_names_run_deck(token, arg) for arg in cmdline)
            if not (candidates & wanted or launched_here):
                continue
            proc.kill()
            killed += 1
            logger.info(
                "Killed simulator process %d (%s) carrying token %s",
                proc.pid,
                proc.info.get("name"),
                token,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed


class ProcessPresence(Enum):
    """Whether the recorded process can still execute work."""

    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


def process_start_marker(pid: int) -> str:
    """Return process-start identity independent of Linux wall-clock corrections.

    Linux's epoch creation time adds the current boot-time estimate to a
    stable kernel tick count. Persist the tick count and boot ID directly.
    Windows records a fixed creation FILETIME, exposed by psutil.
    """
    if sys.platform == "linux":
        proc = Path(psutil.PROCFS_PATH)
        boot = (proc / "sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        stat = (proc / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        # The parenthesized command may contain spaces and closing parens.
        # The fields after its final ')' start at field 3; starttime is 22.
        ticks = int(stat.rpartition(")")[2].split()[19])
        return f"linux:{boot}:{ticks}"
    return "birth:" + psutil.Process(pid).create_time().hex()


def process_identity_presence(pid: int, start_marker: str | None) -> ProcessPresence:
    """Compare a process's start identity as well as its reusable PID."""
    if pid <= 0 or not start_marker:
        return ProcessPresence.UNKNOWN
    try:
        process = psutil.Process(pid)
        if process_start_marker(pid) != start_marker or process.status() == psutil.STATUS_ZOMBIE:
            return ProcessPresence.ABSENT
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return ProcessPresence.ABSENT
    except (psutil.Error, OSError, ValueError, IndexError):
        return ProcessPresence.UNKNOWN
    return ProcessPresence.PRESENT


def _windows_process_name(pid: int) -> str | None:
    """Read an unnamed process's OS name without opening its image or memory.

    Some Windows system entries have no executable path for psutil to name.
    Toolhelp supplies their names. A missing name, failed query, bounded scan
    exhaustion or changed process identity remains unresolved.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    try:
        started = process_start_marker(pid)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        kernel.Process32FirstW.restype = wintypes.BOOL
        kernel.Process32NextW.argtypes = kernel.Process32FirstW.argtypes
        kernel.Process32NextW.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        deadline = time.monotonic() + 2
        snapshot = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in (None, ctypes.c_void_p(-1).value):
            return None
        name = None
        try:
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            for _ in range(65536):
                if not more or time.monotonic() >= deadline:
                    break
                if entry.th32ProcessID == pid:
                    name = entry.szExeFile or None
                    break
                more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            closed = kernel.CloseHandle(snapshot)
        if not closed or process_identity_presence(pid, started) is not ProcessPresence.PRESENT:
            return None
        return name
    except (psutil.Error, OSError, ValueError):
        return None


def simulator_presence(token: str, executable_names: Collection[str]) -> ProcessPresence:
    """Inspect simulator candidates without turning unreadable commands into absence."""
    wanted = {name.lower() for name in executable_names if name}
    if not token or not wanted:
        return ProcessPresence.UNKNOWN
    unknown = False
    try:
        for process in psutil.process_iter(("name", "cmdline", "status"), ad_value=None):
            try:
                if process.info.get("status") == psutil.STATUS_ZOMBIE:
                    continue
                name = process.info.get("name")
                command = process.info.get("cmdline")
                candidates = {name.lower()} if name else set()
                if command:
                    candidates.update(PurePath(arg).name.lower() for arg in command[:2])
                if not candidates and sys.platform == "win32":
                    fallback = _windows_process_name(process.pid)
                    if fallback:
                        candidates.add(fallback.lower())
                if not candidates:
                    unknown = True
                elif candidates & wanted:
                    if not command:
                        unknown = True
                    elif any(token_in_argument(token, arg) for arg in command):
                        return ProcessPresence.PRESENT
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.Error, OSError):
                unknown = True
    except (psutil.Error, OSError):
        unknown = True
    return ProcessPresence.UNKNOWN if unknown else ProcessPresence.ABSENT


def kill_process_group(pid: int, sig: int | None = None) -> bool:
    """Signal the whole process group of a child spawned as a session leader
    (its pid is the group id): the child and everything it started. ``sig``
    defaults to SIGKILL, resolved here because Windows has no such name. True
    when the group was signalled; False on Windows, which has no group to
    signal, and when the group is already gone — the caller then signals the
    one process it holds, if it still needs to.
    """
    if os.name != "posix":
        return False
    try:
        os.killpg(pid, signal.SIGKILL if sig is None else sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True
