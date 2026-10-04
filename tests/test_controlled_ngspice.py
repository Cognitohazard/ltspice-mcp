"""Controlled startup belongs to one launch, never the parent process environment."""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from ltspice_mcp.lib.pdk_native import ArtifactDigest
from ltspice_mcp.lib.recovery_records import ExecutionRecord, StartupPolicy
from ltspice_mcp.lib.simulator_build import SimulatorExecutable


def _execution(folder: Path, mode: str):
    folder.mkdir()
    startup = folder / "spinit"
    startup.write_bytes(b"* Controlled startup\n")
    policy = StartupPolicy(
        "ngspice-empty-init-v1",
        True,
        ArtifactDigest(startup, hashlib.sha256(startup.read_bytes()).hexdigest()),
        (("SPICE_SCRIPTS", str(folder)),),
    )
    return ExecutionRecord(
        None,
        "unbounded",
        2,
        None,
        10,
        ("ngspice",),
        SimulatorExecutable("ngspice", "a" * 64, 1, "2026-10-01"),
        mode,
        "test",
        policy,
    )


def test_concurrent_launches_keep_separate_environments(tmp_path, monkeypatch):
    from ltspice_mcp.lib.controlled_ngspice import controlled_ngspice

    monkeypatch.setenv("SPICE_SCRIPTS", "ambient-script-directory")
    monkeypatch.setenv("SPICE_LIB_DIR", "ambient-model-directory")
    before = dict(os.environ)
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    executions = [_execution(tmp_path / name, mode) for name, mode in [("a", "hsa"), ("b", "ps")]]
    adapters = [controlled_ngspice(execution, lambda: None) for execution in executions]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda adapter: adapter.run(tmp_path / "deck.cir"), adapters))
    assert results == [0, 0]
    assert dict(os.environ) == before
    calls = run.call_args_list
    assert {call.kwargs["env"]["SPICE_SCRIPTS"] for call in calls} == {
        str(tmp_path / "a"),
        str(tmp_path / "b"),
    }
    assert all("SPICE_LIB_DIR" not in call.kwargs["env"] for call in calls)
    assert {call.args[0][call.args[0].index("-D") + 1] for call in calls} == {
        "ngbehavior=hsa",
        "ngbehavior=ps",
    }
    assert all("-n" in call.args[0] for call in calls)


def test_verification_refuses_before_spawn(tmp_path, monkeypatch):
    from ltspice_mcp.lib.controlled_ngspice import controlled_ngspice

    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    def changed():
        raise ValueError("frozen bytes changed")

    adapter = controlled_ngspice(_execution(tmp_path / "startup", "hsa"), changed)
    with pytest.raises(ValueError, match="frozen bytes changed"):
        adapter.run(tmp_path / "deck.cir")
    run.assert_not_called()


def test_real_ngspice_uses_only_the_recorded_startup(tmp_path, monkeypatch):
    import shutil
    from dataclasses import replace

    from ltspice_mcp.lib.controlled_ngspice import controlled_ngspice
    from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead

    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    for filename in ("spinit", ".spiceinit", "spice.rc"):
        (ambient / filename).write_text("echo UNEXPECTED_STARTUP\nquit 19\n")
    monkeypatch.setenv("SPICE_SCRIPTS", str(ambient))
    monkeypatch.setenv("SPICE_USERINIT_DIR", str(ambient))
    policy = _execution(tmp_path / "controlled", "hsa")
    execution = replace(policy, simulator_argv=(executable,))
    deck = tmp_path / "bench.cir"
    deck.write_text("* Controlled startup\nV1 n 0 1\nR1 n 0 1000\n.op\n.end\n")
    checked = []
    adapter = controlled_ngspice(execution, lambda: checked.append(True))
    assert adapter.run(deck, timeout=10, cwd=ambient, exe_log=True) == 0
    assert checked == [True]
    raw = OffsetAwareRawRead(deck.with_suffix(".raw"), dialect="ngspice")
    assert float(raw.get_wave("v(n)")[0]) == pytest.approx(1.0)
    assert "UNEXPECTED_STARTUP" not in deck.with_suffix(".exe.log").read_text(errors="replace")
    assert os.environ["SPICE_SCRIPTS"] == str(ambient)
