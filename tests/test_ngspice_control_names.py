"""How the server writes a file name into an ngspice control script.

Three writers do it: the seeded and native PDK driver (``seeded_commands``),
the ``write`` added to a deck's own ``.control`` block
(``inject_ngspice_control_write``), and the hierarchy check's driver. All of
them write the name through ``ngspice_driver.control_path``. Measured with
ngspice-42: a name in single quotes keeps its spaces, a name in double quotes
keeps the quotes as part of the name, and braces are dropped either way.
"""

from __future__ import annotations

import ast
import itertools
import re
import shutil
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.ngspice_driver import control_path, seeded_commands
from ltspice_mcp.lib.ngspice_mode import run_batch
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import inject_ngspice_control_write
from tests._source_scan import modules
from tests.conftest import terminal_experiment
from tests.test_ngspice_e2e import _signals

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

SCRIPTED = (
    "* rc step via control script\n"
    "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.tran 1u 5m\n"
    ".control\nrun\nquit\n.endc\n.end\n"
)
"""A deck whose .control block runs the analysis and quits, so ngspice-42
writes no raw of its own (measured); the server adds a ``write`` before the
``quit``."""


@pytest.mark.parametrize(
    ("name", "spelled"),
    [
        ("run_7.input.cir", "run_7.input.cir"),
        ("C:/work/run.raw", "C:/work/run.raw"),
        ("with space/deck.cir", "'with space/deck.cir'"),
        ("semi;colon#hash(x)&%~!", "'semi;colon#hash(x)&%~!'"),
        ("ünï/r.raw", "'ünï/r.raw'"),
    ],
)
def test_a_plain_name_is_kept_and_any_other_is_single_quoted(name: str, spelled: str):
    assert control_path(name) == spelled


@pytest.mark.parametrize(
    "name", ["", "it's", 'say "x"', "$HOME", "a`b`", "a\\b", "br{ace}", "a\tb", "a\nb", "a\x00b"]
)
def test_a_name_the_control_language_would_change_is_refused(name: str):
    with pytest.raises(ValueError, match="no ngspice control-script spelling"):
        control_path(name)


def test_a_plain_seeded_driver_keeps_its_bytes():
    """Recorded native and seeded drivers are verified byte for byte."""
    assert seeded_commands(17, "run_7.input.cir", "run_7.raw") == (
        "setseed 17\nsource run_7.input.cir\nrun\nwrite run_7.raw\nquit\n"
    )


@needs_ngspice
def test_a_seeded_driver_reads_and_writes_names_with_spaces(tmp_path: Path):
    (tmp_path / "in dir").mkdir()
    (tmp_path / "out dir").mkdir()
    (tmp_path / "in dir" / "deck.cir").write_text(
        "* divider\nV1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.op\n.end\n", encoding="utf-8"
    )
    driver = (
        "* seeded\n.control\n"
        + seeded_commands(17, "in dir/deck.cir", "out dir/run.raw")
        + ".endc\n.end\n"
    )
    log = run_batch(
        tuple(NGspiceSimulator.spice_exe),
        "",
        driver.encode("utf-8"),
        cwd=tmp_path,
        clean_startup=True,
        timeout=60,
    )
    assert (tmp_path / "out dir" / "run.raw").stat().st_size > 0, log


def test_a_write_into_a_folder_ngspice_would_change_is_not_added(tmp_path: Path):
    deck = tmp_path / "ctrl.cir"
    deck.write_text(SCRIPTED, encoding="utf-8")
    folder = tmp_path / "br{ace}"
    folder.mkdir()
    assert inject_ngspice_control_write(deck, NGspiceSimulator, "job1", folder) == deck


@needs_ngspice
async def test_a_scripted_run_in_a_folder_with_a_space_has_a_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A scripted deck gets its raw from the added ``write``, which must
    reach a run folder whose path has a space, as Windows profiles often do."""
    work_dir = tmp_path / "work dir"
    work_dir.mkdir()
    monkeypatch.setattr(
        NGspiceSimulator, "_compatibility_mode", NGspiceSimulator._compatibility_mode
    )
    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    state = SessionState.create(config, detect_simulators(config))
    deck = work_dir / "ctrl_step.cir"
    deck.write_text(SCRIPTED, encoding="utf-8")
    receipt = await terminal_experiment(
        state,
        {
            "request_id": "scripted-spaced",
            "circuits": [{"path": str(deck)}],
            "execution": {"simulator": "ngspice", "wait_s": 90},
        },
    )
    assert receipt["status"] == "completed", receipt
    signals = await _signals(state, receipt["job_id"])
    assert any(signal.lower() == "v(out)" for signal in signals), signals


#: Every function that writes a file name after ``source`` or ``write`` in
#: an ngspice control script. Each spells it with ``control_path``.
CONTROL_NAME_WRITERS = {
    ("lib/ngspice_driver.py", "seeded_commands"),
    ("lib/hierarchy_check.py", "_driver"),
    ("lib/runner_base.py", "inject_ngspice_control_write"),
}

_COMMAND = re.compile(r"(?:^|\n)(?:source|write) $")


def _functions(tree: ast.AST, prefix: str = "") -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = f"{prefix}{node.name}"
            found.append((name, node))
            found.extend(_functions(node, f"{name}."))
        elif isinstance(node, ast.ClassDef):
            found.extend(_functions(node, f"{prefix}{node.name}."))
    return found


def _writes_a_control_name(node: ast.AST) -> bool:
    """An f-string with a ``source``/``write`` command whose argument is
    interpolated."""
    if not isinstance(node, ast.JoinedStr):
        return False
    parts = node.values
    return any(
        isinstance(part, ast.Constant)
        and isinstance(part.value, str)
        and _COMMAND.search(part.value)
        and isinstance(following, ast.FormattedValue)
        for part, following in itertools.pairwise(parts)
    )


def test_every_control_script_name_is_spelled_by_control_path():
    writers = set()
    for path, tree in modules().items():
        for name, function in _functions(tree):
            if not any(
                _writes_a_control_name(node)
                for node in ast.walk(function)
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or node is function
            ):
                continue
            writers.add((path, name))
            calls = {
                call.func.id
                for call in ast.walk(function)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            assert "control_path" in calls, f"{path}:{name} writes a name without control_path"
    assert writers == CONTROL_NAME_WRITERS
