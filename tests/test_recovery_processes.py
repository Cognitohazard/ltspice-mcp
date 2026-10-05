"""Recovery needs positive evidence that a prior simulator cannot still run."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from ltspice_mcp.lib import proc_kill, wsl


@pytest.mark.parametrize(
    ("processes", "expected"),
    [
        ([], "absent"),
        ([("ngspice", ["ngspice", "exp_case_1.cir"])], "present"),
        ([("ngspice", ["ngspice", "exp_case_10.cir"])], "absent"),
        ([("ngspice", None)], "unknown"),
        ([("editor", None)], "absent"),
        ([(None, None)], "unknown"),
        ([("wrapper", ["wine", "LTspice.exe", "exp_case_1.net"])], "present"),
        ([("ngspice", None), ("ngspice", ["ngspice", "exp_case_1.cir"])], "present"),
    ],
)
@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_local_simulator_presence(monkeypatch, processes, expected, platform):
    # Windows reads a nameless process's image name by pid, so that branch is
    # exercised everywhere; these fake pids name no real process.
    monkeypatch.setattr(proc_kill.sys, "platform", platform)
    monkeypatch.setattr(proc_kill, "_windows_process_name", lambda pid: None)
    entries = [
        SimpleNamespace(pid=1000 + index, info={"name": name, "cmdline": cmd})
        for index, (name, cmd) in enumerate(processes)
    ]
    monkeypatch.setattr(proc_kill.psutil, "process_iter", lambda *a, **kw: iter(entries))
    result = proc_kill.simulator_presence("exp_case_1", {"ngspice", "ltspice.exe"})
    assert result.value == expected


@pytest.mark.parametrize(
    ("platform", "image_name", "expected"),
    [
        # A nameless process stays unknown off Windows: nothing else names it.
        ("linux", "notepad.exe", "unknown"),
        # On Windows its image name settles it: another program is absence,
        # a simulator with no readable command line is still unknown.
        ("win32", "notepad.exe", "absent"),
        ("win32", "LTspice.exe", "unknown"),
        ("win32", None, "unknown"),
    ],
)
def test_windows_names_a_nameless_process_by_pid(monkeypatch, platform, image_name, expected):
    monkeypatch.setattr(proc_kill.sys, "platform", platform)
    looked_up: list[int] = []

    def image(pid):
        looked_up.append(pid)
        return image_name

    monkeypatch.setattr(proc_kill, "_windows_process_name", image)
    nameless = SimpleNamespace(pid=4242, info={"name": None, "cmdline": None})
    monkeypatch.setattr(proc_kill.psutil, "process_iter", lambda *a, **kw: iter([nameless]))

    result = proc_kill.simulator_presence("exp_case_1", {"ngspice", "ltspice.exe"})

    assert result.value == expected
    assert looked_up == ([4242] if platform == "win32" else [])


def test_presence_query_failure_is_not_absence(monkeypatch):
    def denied(*args, **kwargs):
        raise psutil.AccessDenied()

    monkeypatch.setattr(proc_kill.psutil, "process_iter", denied)
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice"}).value == "unknown"


def test_exited_zombie_simulator_cannot_hold_recovery(monkeypatch):
    zombie = SimpleNamespace(
        info={"name": "ngspice", "cmdline": None, "status": psutil.STATUS_ZOMBIE}
    )
    monkeypatch.setattr(proc_kill.psutil, "process_iter", lambda *a, **kw: iter([zombie]))
    assert proc_kill.simulator_presence("exp_case_1", {"ngspice"}).value == "absent"


def test_owner_identity_includes_creation_time():
    started = proc_kill.process_start_marker(os.getpid())
    assert proc_kill.process_identity_presence(os.getpid(), started).value == "present"
    assert (
        proc_kill.process_identity_presence(os.getpid(), started + "different").value == "absent"
    )
    assert proc_kill.process_identity_presence(os.getpid(), None).value == "unknown"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux boot-time clock correction")
def test_owner_identity_survives_wall_clock_correction(monkeypatch):
    from psutil import _pslinux

    started = proc_kill.process_start_marker(os.getpid())
    original = _pslinux.boot_time()
    monkeypatch.setattr(_pslinux, "boot_time", lambda: original + 15)
    assert proc_kill.process_identity_presence(os.getpid(), started).value == "present"


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected"),
    [
        ("[]", 0, "absent"),
        (
            json.dumps([{"Name": "LTspice.exe", "CommandLine": "LTspice.exe exp_case_1.cir"}]),
            0,
            "present",
        ),
        (
            json.dumps([{"Name": "LTspice.exe", "CommandLine": "LTspice.exe exp_case_10.cir"}]),
            0,
            "absent",
        ),
        (json.dumps([{"Name": "LTspice.exe", "CommandLine": None}]), 0, "unknown"),
        ("[]", 1, "unknown"),
        ("not json", 0, "unknown"),
        ("", 0, "unknown"),
    ],
)
def test_wsl_presence_retains_unreadable_candidates(monkeypatch, stdout, returncode, expected):
    monkeypatch.setattr(wsl, "is_wsl", lambda: True)
    run = Mock(return_value=subprocess.CompletedProcess([], returncode, stdout, ""))
    monkeypatch.setattr(wsl.subprocess, "run", run)
    assert wsl.windows_ltspice_presence("exp_case_1").value == expected
    script = run.call_args.args[0][-1]
    assert "Where-Object" not in script
    assert "taskkill" not in script


def test_wsl_presence_is_bounded_and_read_only(monkeypatch):
    monkeypatch.setattr(wsl, "is_wsl", lambda: True)
    run = Mock(side_effect=subprocess.TimeoutExpired("powershell", 1))
    monkeypatch.setattr(wsl.subprocess, "run", run)
    assert wsl.windows_ltspice_presence("exp_case_1").value == "unknown"
    assert run.call_args.kwargs["timeout"] <= 15
    run.reset_mock()
    assert wsl.windows_ltspice_presence("bad;token").value == "unknown"
    run.assert_not_called()
