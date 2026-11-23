"""Which environment a simulator process is started with.

Some MCP clients put ``NoDefaultCurrentDirectoryInExePath`` in the
environment of the servers they start, and LTspice on macOS, started through
a Wine launcher, then writes no output. Every launch of a simulator is
started without it on macOS and under Wine, and the server's own environment
keeps it.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest

from tests._source_scan import modules, scopes

VARIABLE = "NoDefaultCurrentDirectoryInExePath"

needs_posix_script = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake simulator is a POSIX shell script"
)


@pytest.fixture
def wine_host(monkeypatch):
    """A host that starts LTspice under Wine or on macOS: not Windows, not
    WSL. The client's variable is set in the server's environment."""
    from ltspice_mcp.lib import wsl

    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    monkeypatch.setenv(VARIABLE, "1")
    monkeypatch.setenv("LTSPICE_MCP_TEST_KEPT", "kept")


@pytest.fixture
def fake_simulator(tmp_path):
    """An executable that records the environment it was started with in
    ``<itself>.env`` and exits 0. Its name holds "wine", so spicelib's LTspice
    takes it for a Wine launcher on macOS too."""
    script = tmp_path / "wine-fake-simulator"
    script.write_text('#!/bin/sh\nenv > "$0.env"\nexit 0\n', encoding="utf-8")
    script.chmod(0o755)
    return script


def _recorded(script: Path) -> dict[str, str]:
    text = (script.parent / f"{script.name}.env").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _deck(tmp_path: Path) -> Path:
    deck = tmp_path / "rc.cir"
    deck.write_text("* RC\nR1 in 0 1k\n.op\n.end\n", encoding="utf-8")
    return deck


_SPICELIB_FAMILIES = {
    "ltspice": ("spicelib.simulators.ltspice_simulator", "LTspice"),
    "ngspice": ("spicelib.simulators.ngspice_simulator", "NGspiceSimulator"),
    "qspice": ("spicelib.simulators.qspice_simulator", "Qspice"),
    "xyce": ("spicelib.simulators.xyce_simulator", "XyceSimulator"),
}


@needs_posix_script
@pytest.mark.parametrize("family", sorted(_SPICELIB_FAMILIES))
def test_spicelib_run_starts_without_the_variable(wine_host, fake_simulator, tmp_path, family):
    import importlib

    import ltspice_mcp.lib.simulator  # noqa: F401 - the server's own import

    module, name = _SPICELIB_FAMILIES[family]
    simulator = getattr(importlib.import_module(module), name).create_from(str(fake_simulator))
    simulator.run(_deck(tmp_path), timeout=30)
    started = _recorded(fake_simulator)
    assert VARIABLE not in started
    assert started["LTSPICE_MCP_TEST_KEPT"] == "kept"
    assert os.environ[VARIABLE] == "1"


@needs_posix_script
def test_spicelib_ltspice_netlist_export_starts_without_the_variable(
    wine_host, fake_simulator, tmp_path
):
    from ltspice_mcp.lib.simulator import _get_ltspice_class

    sheet = tmp_path / "rc.asc"
    sheet.write_text("Version 4\nSHEET 1 880 680\n", encoding="utf-8")
    simulator = _get_ltspice_class().create_from(str(fake_simulator))
    with pytest.raises(RuntimeError, match="Failed to create netlist"):  # the fake writes none
        simulator.create_netlist(sheet)
    started = _recorded(fake_simulator)
    assert VARIABLE not in started
    assert started["LTSPICE_MCP_TEST_KEPT"] == "kept"


@pytest.mark.parametrize("server_first", [False, True], ids=["spicelib-first", "server-first"])
def test_every_spicelib_module_launches_through_run_simulator(server_first):
    """Every spicelib module that holds a ``run_function`` holds
    ``run_simulator``, whether it was loaded before the server's modules or
    after them."""
    script = f"""
import importlib, pkgutil, sys

def load_spicelib():
    import spicelib
    for info in pkgutil.walk_packages(spicelib.__path__, "spicelib."):
        try:
            importlib.import_module(info.name)
        except Exception:
            pass

if {server_first!r}:
    import ltspice_mcp.lib.simulator
    load_spicelib()
else:
    load_spicelib()
    import ltspice_mcp.lib.simulator

from ltspice_mcp.lib.simulator_env import run_simulator
holders = [
    name for name, module in sorted(sys.modules.items())
    if name.startswith("spicelib") and hasattr(module, "run_function")
]
assert len(holders) >= 6, holders
print("\\n".join(
    name for name in holders if sys.modules[name].run_function is not run_simulator
))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"spicelib modules that bypass run_simulator: {result.stdout}"


@needs_posix_script
def test_question_to_ngspice_starts_without_the_variable(wine_host, fake_simulator):
    from ltspice_mcp.lib.ngspice_mode import run_batch

    # The fake writes no log, so the question itself fails after the start.
    with pytest.raises(FileNotFoundError):
        run_batch((str(fake_simulator),), "", b"* probe\n.end\n", clean_startup=False, timeout=30)
    started = _recorded(fake_simulator)
    assert VARIABLE not in started
    assert started["LTSPICE_MCP_TEST_KEPT"] == "kept"


@needs_posix_script
def test_version_probe_starts_without_the_variable(wine_host, fake_simulator):
    from ltspice_mcp.lib.native_execution import observe_simulator

    class Probed:
        spice_exe: ClassVar[list[str]] = [str(fake_simulator)]

    observe_simulator(Probed)
    started = _recorded(fake_simulator)
    assert VARIABLE not in started
    assert started["LTSPICE_MCP_TEST_KEPT"] == "kept"


def test_controlled_ngspice_starts_without_the_variable(wine_host, tmp_path, monkeypatch):
    from ltspice_mcp.lib import simulator_env
    from ltspice_mcp.lib.controlled_ngspice import controlled_ngspice

    monkeypatch.setattr(simulator_env, "_host_removes", lambda: True)
    from tests.test_controlled_ngspice import _execution

    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    adapter = controlled_ngspice(_execution(tmp_path / "startup", "hsa"), lambda: None)
    assert adapter.run(tmp_path / "deck.cir") == 0
    env = run.call_args.kwargs["env"]
    assert VARIABLE not in env
    assert env["LTSPICE_MCP_TEST_KEPT"] == "kept"
    assert env["SPICE_SCRIPTS"] == str(tmp_path / "startup")


@pytest.mark.parametrize(
    ("platform", "wsl", "removed"),
    [
        ("darwin", False, True),
        ("linux", False, True),
        ("linux", True, False),
        ("win32", False, False),
    ],
    ids=["macos", "linux-wine", "wsl-interop", "windows"],
)
def test_the_variable_is_removed_only_on_macos_and_under_wine(monkeypatch, platform, wsl, removed):
    from ltspice_mcp.lib import simulator_env
    from ltspice_mcp.lib import wsl as wsl_module

    monkeypatch.setattr(simulator_env, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(wsl_module, "is_wsl", lambda: wsl)
    base = {"nodefaultcurrentdirectoryinexepath": "1", "PATH": "/bin"}
    env = simulator_env.simulator_environment(base)
    assert ("nodefaultcurrentdirectoryinexepath" not in env) is removed
    assert env["PATH"] == "/bin"
    assert base == {"nodefaultcurrentdirectoryinexepath": "1", "PATH": "/bin"}


# --- Every launch decides its environment ------------------------------------

#: The modules that start a simulator. Each launch in them passes ``env=``,
#: built by ``simulator_env.simulator_environment``.
SIMULATOR_LAUNCHERS = {
    "lib/controlled_ltspice.py",
    "lib/controlled_ngspice.py",
    "lib/ltspice_bridge.py",
    "lib/ltspice_windows.py",
    "lib/ltspice_wsl.py",
    "lib/native_execution.py",
    "lib/ngspice_mode.py",
    "lib/simulator_env.py",
}

#: Launches elsewhere in the package that start no simulator, by module and
#: scope, with why.
OTHER_LAUNCHES = {
    ("api/_detach.py", "submit"): "starts the server's own detached-job owner",
    ("api/_detach.py", "_stop_owner"): "stops the detached-job owner",
    ("lib/desktop.py", "_open_app_window"): "opens a plot page in a browser window",
    ("lib/desktop.py", "open_in_desktop"): "opens a plot page with the desktop's opener",
    ("lib/parser_bootstrap.py", "_guardian"): "starts the parser worker",
    ("lib/parser_process.py", "ParserTree.__init__"): "starts the parser worker",
    ("lib/wsl.py", "to_windows_path"): "asks wslpath to convert a path",
    ("lib/wsl.py", "_resolve_win_env"): "asks cmd for a Windows variable",
    ("lib/wsl.py", "windows_ltspice_presence"): "asks tasklist which processes run",
    ("lib/wsl.py", "kill_windows_ltspice_by_token"): "stops LTspice with taskkill",
    ("tools/run_code.py", "CodeWorker._spawn"): "starts the run_code worker",
}

#: Calls that start a process, by the name they are imported under.
_LAUNCH_FUNCTIONS = {
    *(f"subprocess.{name}" for name in ("run", "Popen", "call", "check_call", "check_output")),
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    *(f"os.{name}" for name in ("system", "popen", "startfile", "posix_spawn", "posix_spawnp")),
    *(f"os.spawn{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
    *(f"os.exec{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
}

#: Bare names that start a process: the hidden-desktop launch, a local
#: ``spawn`` that stands for ``subprocess.Popen``, and spicelib's launcher.
_LAUNCH_NAMES = {"run_on_desktop", "spawn", "run_function"}


def _aliases(tree: ast.Module) -> dict[str, str]:
    """Each local name an import binds, mapped to the dotted name it stands
    for: ``import subprocess as sp`` binds ``sp`` to ``subprocess``, ``from os
    import system`` binds ``system`` to ``os.system``."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _is_desktop_launch(func: ast.Attribute) -> bool:
    """``<desktop>.start(...)`` or ``<desktop>.run(...)`` on a hidden desktop."""
    if func.attr not in {"start", "run"}:
        return False
    owner = func.value
    name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
    return "desktop" in name.lower()


def _launches(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    """Each call in ``tree`` that starts a process, with its scope."""
    bound = _aliases(tree)
    found = []
    for scope, nodes in scopes(tree):
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                owner = bound.get(func.value.id, func.value.id)
                launch = f"{owner}.{func.attr}" in _LAUNCH_FUNCTIONS or _is_desktop_launch(func)
            elif isinstance(func, ast.Attribute):
                launch = _is_desktop_launch(func)
            elif isinstance(func, ast.Name):
                launch = (
                    func.id in _LAUNCH_NAMES
                    or bound.get(func.id, "") in _LAUNCH_FUNCTIONS
                    or bound.get(func.id, "").endswith(".run_function")
                )
            else:
                launch = False
            if launch:
                found.append((scope, node))
    return found


def test_the_scan_finds_each_form_of_a_launch():
    source = """
import subprocess as sp
from os import system
from asyncio import create_subprocess_shell
from spicelib.sim.simulator import run_function as launch

def f(desktop):
    sp.check_output(["a"])
    system("a")
    create_subprocess_shell("a")
    launch(["a"])
    desktop.start(["a"])
    self.desktop.run(["a"])
    tasks.spawn(work)
"""
    calls = sorted(call.lineno for _, call in _launches(ast.parse(source)))
    assert calls == [8, 9, 10, 11, 12, 13]


def test_every_process_launch_is_a_simulator_launch_or_named():
    unexpected = sorted(
        f"{path}:{call.lineno} ({scope})"
        for path, tree in modules().items()
        if path not in SIMULATOR_LAUNCHERS
        for scope, call in _launches(tree)
        if (path, scope) not in OTHER_LAUNCHES
    )
    assert unexpected == [], (
        "a process is started outside the scopes this test knows; a simulator "
        "launch goes in SIMULATOR_LAUNCHERS and passes env=simulator_environment(), "
        "any other launch goes in OTHER_LAUNCHES with why"
    )


def test_every_named_other_launch_still_launches():
    found = {
        (path, scope) for path, tree in modules().items() for scope, _ in _launches(tree)
    }
    assert set(OTHER_LAUNCHES) - found == set()


def _passes_an_environment(call: ast.Call) -> bool:
    return any(
        keyword.arg == "env"
        and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
        for keyword in call.keywords
    )


def test_every_simulator_launch_passes_an_environment():
    missing = sorted(
        f"{path}:{call.lineno} ({scope})"
        for path, tree in modules().items()
        if path in SIMULATOR_LAUNCHERS
        for scope, call in _launches(tree)
        if not _passes_an_environment(call)
    )
    assert missing == [], (
        "a simulator launch takes the server's environment instead of simulator_environment()"
    )


def test_only_simulator_env_reads_the_whole_server_environment_for_a_launch():
    readers = []
    for path, tree in modules().items():
        if path not in SIMULATOR_LAUNCHERS or path == "lib/simulator_env.py":
            continue
        bound = _aliases(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "environ"
                and isinstance(node.value, ast.Name)
                and bound.get(node.value.id, node.value.id) == "os"
            ) or (isinstance(node, ast.Name) and bound.get(node.id) == "os.environ"):
                readers.append(f"{path}:{node.lineno}")
    assert readers == [], "a simulator launch builds its environment from simulator_environment()"
