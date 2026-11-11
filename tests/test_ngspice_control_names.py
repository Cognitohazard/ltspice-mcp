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
import sys
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
from tests.conftest import needs_controlled_startup, terminal_experiment
from tests.test_ngspice_e2e import _signals
from tests.test_resume_surface import _state as _seeded_state

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
    ],
)
def test_a_plain_name_is_kept_and_any_other_is_single_quoted(name: str, spelled: str):
    assert control_path(name) == spelled


@pytest.mark.parametrize(
    "name",
    [
        "",
        "it's",
        'say "x"',
        "$HOME",
        "a`b`",
        "a\\b",
        "br{ace}",
        "a\tb",
        "a\x0bb",
        "a\x0cb",
        "a\nb",
        "a\x00b",
        "~/r.raw",
    ],
)
def test_a_name_the_control_language_would_change_is_refused(name: str):
    with pytest.raises(ValueError, match="cannot keep the file name"):
        control_path(name)


def test_a_non_ascii_name_is_written_only_into_a_utf8_script_off_windows():
    """ngspice-42 stops on a deck that is not valid UTF-8; how ngspice on
    Windows reads a non-ASCII name is not measured."""
    for encoding in ("cp1252", "ascii"):
        with pytest.raises(ValueError, match="is not ASCII"):
            control_path("José/r.raw", encoding)
    if sys.platform == "win32":
        with pytest.raises(ValueError, match="is not ASCII"):
            control_path("José/r.raw", "utf-8")
    else:
        assert control_path("José/r.raw", "utf-8") == "'José/r.raw'"


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


def test_a_cp1252_deck_gets_no_write_to_a_folder_name_it_cannot_carry(tmp_path: Path):
    """The write would be cp1252 bytes in a deck ngspice-42 reads as UTF-8,
    and it stops on them ("UTF-8 syntax error in input deck")."""
    deck = tmp_path / "ctrl.cir"
    deck.write_bytes(SCRIPTED.replace("* rc step", "* rc step, 5 µs").encode("cp1252"))
    folder = tmp_path / "José dir"
    folder.mkdir()
    assert inject_ngspice_control_write(deck, NGspiceSimulator, "job1", folder) == deck


@needs_ngspice
@needs_controlled_startup
async def test_a_seeded_run_of_a_deck_in_a_folder_with_a_space_completes(tmp_path: Path):
    """The seeded driver sources the deck by its path under the job's
    folder, which keeps the deck's own folder name."""
    folder = tmp_path / "deck dir"
    (folder / "models with spaces").mkdir(parents=True)
    (folder / "models with spaces" / "draw.inc").write_bytes(b".param draw=agauss(10,1,1)\n")
    deck = folder / "draw.cir"
    deck.write_bytes(
        b'* seeded draw\n.include "models with spaces/draw.inc"\n'
        b"V1 n 0 {draw}\nR1 n 0 1000\n.op\n.end\n"
    )
    receipt = await terminal_experiment(
        _seeded_state(tmp_path),
        {
            "request_id": "seeded-spaced",
            "circuits": [{"path": str(deck)}],
            "execution": {"recoverable": True, "simulator_seed": 17, "wait_s": 90},
            "lint": "off",
        },
    )
    assert receipt["status"] == "completed", receipt


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


#: Every function that writes a file name into an ngspice control script.
#: Each spells it with ``control_path``.
CONTROL_NAME_WRITERS = {
    ("lib/ngspice_driver.py", "seeded_commands"),
    ("lib/hierarchy_check.py", "_driver"),
    ("lib/runner_base.py", "inject_ngspice_control_write"),
}

_COMMANDS = "source|write|wrdata|load|codemodel|pre_codemodel|osdi|pre_osdi"
_COMMAND_BEFORE_SLOT = re.compile(rf"(?:^|\n)[ \t]*(?:{_COMMANDS})[ \t]+['\"]?$")
"""A command at the start of a line, with only its file name still to come."""
_COMMAND_WITH_PLACEHOLDER = re.compile(
    rf"(?:^|\n)[ \t]*(?:{_COMMANDS})[ \t]+['\"]?(?:%[sr]|\{{[^{{}}]*\}})"
)
"""A command whose file name is a ``%`` or ``.format`` placeholder."""


def _hits(tree: ast.AST, scope: str = "<module>"):
    """Each place in ``tree`` that writes a control-script file name, with
    the innermost function it is in and the value written, None when it is
    not an f-string."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inner = node.name if scope == "<module>" else f"{scope}.{node.name}"
            yield from _hits(node, inner)
            continue
        if isinstance(node, ast.ClassDef):
            yield from _hits(node, scope)
            continue
        if isinstance(node, ast.JoinedStr):
            parts = node.values
            for part, following in itertools.pairwise(parts):
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and _COMMAND_BEFORE_SLOT.search(part.value)
                ):
                    yield scope, following
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _COMMAND_BEFORE_SLOT.search(node.value) or _COMMAND_WITH_PLACEHOLDER.search(
                node.value
            ):
                yield scope, None
            continue
        yield from _hits(node, scope)


def _from_control_path(value: ast.AST | None, function: ast.AST | None) -> bool:
    """Whether ``value`` is a call of ``control_path`` or a name the function
    assigns from one."""
    if not isinstance(value, ast.FormattedValue):
        return False
    expression = value.value
    if isinstance(expression, ast.Call):
        return isinstance(expression.func, ast.Name) and expression.func.id == "control_path"
    if not isinstance(expression, ast.Name) or function is None:
        return False
    return any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "control_path"
        and any(isinstance(t, ast.Name) and t.id == expression.id for t in node.targets)
        for node in ast.walk(function)
    )


def _function(tree: ast.AST, qualified: str) -> ast.AST | None:
    node: ast.AST | None = tree
    for name in qualified.split("."):
        node = next(
            (
                child
                for child in ast.walk(node)
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == name
            ),
            None,
        )
        if node is None:
            return None
    return node


def control_name_writers(sources: dict[str, ast.Module]) -> tuple[set, list[str]]:
    """The functions that write a control-script file name, and each place
    that does not take it from ``control_path``."""
    writers: set[tuple[str, str]] = set()
    unspelled: list[str] = []
    for path, tree in sources.items():
        for scope, value in _hits(tree):
            writers.add((path, scope))
            function = None if scope == "<module>" else _function(tree, scope)
            if not _from_control_path(value, function):
                unspelled.append(f"{path}:{scope}")
    return writers, unspelled


def test_every_control_script_name_is_spelled_by_control_path():
    writers, unspelled = control_name_writers(modules())
    assert not unspelled, f"names written without control_path: {unspelled}"
    assert writers == CONTROL_NAME_WRITERS


@pytest.mark.parametrize(
    ("snippet", "found"),
    [
        ("def f(p):\n    return f\"source '{p}'\"\n", "f"),
        ("def f(p):\n    return f'write \"{p}\"'\n", "f"),
        ("def f(p):\n    return 'source ' + p\n", "f"),
        ("def f(p):\n    return 'write %s' % p\n", "f"),
        ("def f(p):\n    return 'load {}'.format(p)\n", "f"),
        ("def f(p):\n    return '\\n'.join(['  codemodel ' + p])\n", "f"),
        ("def f(p):\n    def g():\n        return f'wrdata {p}'\n    return g\n", "f.g"),
        ("SCRIPT = '.control\\npre_osdi %s\\n.endc'\n", "<module>"),
    ],
)
def test_the_scan_finds_a_name_written_without_control_path(snippet: str, found: str):
    writers, unspelled = control_name_writers({"probe.py": ast.parse(snippet)})
    assert writers == {("probe.py", found)}
    assert unspelled == [f"probe.py:{found}"]
