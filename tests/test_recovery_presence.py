"""Unreadable process entries require independent names and stable identity."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from ltspice_mcp.lib import proc_kill


def _entry(name=None, command=None, *, pid=77):
    return SimpleNamespace(pid=pid, info={"name": name, "cmdline": command})


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [("Secure System", "absent"), ("ngspice.exe", "unknown"), (None, "unknown"), ("", "unknown")],
)
def test_independent_windows_name_preserves_candidate_guard(monkeypatch, fallback, expected):
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: iter([_entry()]))
    query = Mock(return_value=fallback)
    monkeypatch.setattr(proc_kill, "_windows_process_name", query, raising=False)
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice.exe"}).value == expected
    query.assert_called_once_with(77)


def test_independent_name_cannot_hide_another_unknown_entry(monkeypatch):
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        psutil, "process_iter", lambda *a, **kw: iter([_entry(pid=77), _entry(pid=78)])
    )
    monkeypatch.setattr(
        proc_kill,
        "_windows_process_name",
        Mock(side_effect=["Secure System", None]),
        raising=False,
    )
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice.exe"}).value == "unknown"


def test_independent_name_cannot_hide_a_matching_simulator(monkeypatch):
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform="win32"))
    entries = [_entry(), _entry("ngspice.exe", ["ngspice.exe", "exp_case_1.cir"], pid=78)]
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: iter(entries))
    monkeypatch.setattr(
        proc_kill, "_windows_process_name", Mock(return_value="Secure System"), raising=False
    )
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice.exe"}).value == "present"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_unknown_entries_keep_the_existing_guard(monkeypatch, platform):
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: iter([_entry()]))
    query = Mock(side_effect=AssertionError("Windows lookup on another platform"))
    monkeypatch.setattr(proc_kill, "_windows_process_name", query, raising=False)
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice.exe"}).value == "unknown"
    query.assert_not_called()


def test_readable_name_never_uses_windows_fallback(monkeypatch):
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: iter([_entry("editor")]))
    query = Mock(side_effect=AssertionError("Known name must use existing classification"))
    monkeypatch.setattr(proc_kill, "_windows_process_name", query, raising=False)
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice.exe"}).value == "absent"
    query.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 process snapshot")
def test_real_windows_snapshot_identifies_current_process():
    assert proc_kill._windows_process_name(os.getpid()) == psutil.Process().name()


def _toolhelp(
    monkeypatch,
    *,
    name="Secure System",
    snapshot: int | None = 123,
    identity="present",
    identity_query: Mock | None = None,
):
    import ctypes

    def first(handle, pointer):
        assert pointer._obj.dwSize == ctypes.sizeof(pointer._obj)
        pointer._obj.th32ProcessID = 77
        pointer._obj.szExeFile = name
        return True

    kernel = SimpleNamespace(
        CreateToolhelp32Snapshot=Mock(return_value=snapshot),
        Process32FirstW=Mock(side_effect=first),
        Process32NextW=Mock(return_value=False),
        CloseHandle=Mock(return_value=True),
    )
    monkeypatch.setattr(proc_kill, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(proc_kill, "process_start_marker", Mock(return_value="birth:original"))
    if identity_query is None:
        identity_query = Mock(return_value=proc_kill.ProcessPresence(identity))
    monkeypatch.setattr(proc_kill, "process_identity_presence", identity_query)
    return kernel


def test_windows_name_query_checks_identity_and_closes_snapshot(monkeypatch):
    identity_query = Mock(return_value=proc_kill.ProcessPresence.PRESENT)
    kernel = _toolhelp(monkeypatch, identity_query=identity_query)
    assert proc_kill._windows_process_name(77) == "Secure System"
    kernel.CreateToolhelp32Snapshot.assert_called_once_with(2, 0)
    kernel.CloseHandle.assert_called_once_with(123)
    identity_query.assert_called_once_with(77, "birth:original")


@pytest.mark.parametrize("identity", ["absent", "unknown"])
def test_windows_name_query_rejects_changed_or_unreadable_identity(monkeypatch, identity):
    kernel = _toolhelp(monkeypatch, identity=identity)
    assert proc_kill._windows_process_name(77) is None
    kernel.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("name", ["", "ngspice.exe"])
def test_windows_name_query_does_not_filter_candidate_names(monkeypatch, name):
    _toolhelp(monkeypatch, name=name)
    assert proc_kill._windows_process_name(77) == (name or None)


def test_windows_name_query_failed_snapshot_remains_unresolved(monkeypatch):
    import ctypes

    kernel = _toolhelp(monkeypatch, snapshot=ctypes.c_void_p(-1).value)
    assert proc_kill._windows_process_name(77) is None
    kernel.Process32FirstW.assert_not_called()
    kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("error", [psutil.AccessDenied(), OSError("snapshot unreadable")])
def test_windows_name_query_failure_closes_snapshot(monkeypatch, error):
    kernel = _toolhelp(monkeypatch)
    kernel.Process32FirstW.side_effect = error
    assert proc_kill._windows_process_name(77) is None
    kernel.CloseHandle.assert_called_once_with(123)


def test_windows_name_query_scan_deadline_remains_unresolved(monkeypatch):
    kernel = _toolhelp(monkeypatch)
    monkeypatch.setattr(proc_kill, "time", SimpleNamespace(monotonic=Mock(side_effect=[0, 3])))
    assert proc_kill._windows_process_name(77) is None
    kernel.Process32NextW.assert_not_called()
    kernel.CloseHandle.assert_called_once_with(123)


def test_windows_name_query_close_failure_remains_unresolved(monkeypatch):
    kernel = _toolhelp(monkeypatch)
    kernel.CloseHandle.return_value = False
    assert proc_kill._windows_process_name(77) is None
