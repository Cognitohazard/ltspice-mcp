"""A log's ``.step`` lines are read one way, for the log and for its raw.

``log_parser.read_step_line`` is the one reader: the log's step rows and the
step values the raw decoder binds to each sweep both come from it, so the two
cannot name a step differently.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_parser import StepParameter, parse_step_iterations, read_step_line
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import FIXTURES

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"


@pytest.mark.parametrize(
    ("line", "parameters", "only_parameters"),
    [
        (".step r=1000", [("r", "1000", 1000.0)], True),
        (".step r=1000, c=1e-07", [("r", "1000", 1000.0), ("c", "1e-07", 1e-7)], True),
        # The two builds' stepped temperature, the unit after the number.
        (".step temp=-40°", [("temp", "-40°", -40.0)], True),
        (".step temp=-40°C", [("temp", "-40°C", -40.0)], True),
        ("  .STEP r=5k  ", [("r", "5k", None)], True),
        (".step x=1 and more", [("x", "1", 1.0)], False),
        ("step r=1", None, None),
        ("Measurement: x", None, None),
    ],
)
def test_a_step_line_reads_as_its_parameters(
    line: str,
    parameters: list[tuple[str, str, float | None]] | None,
    only_parameters: bool | None,
):
    step = read_step_line(line)
    if parameters is None:
        assert step is None
        return
    assert step is not None
    assert step.parameters == tuple(StepParameter(*parameter) for parameter in parameters)
    assert step.only_parameters is only_parameters


def _stepped_sweeps() -> list[tuple[str, str]]:
    runs = []
    for build in rec.BUILDS:
        for case_id, entry in sorted(rec.manifest(build)["cases"].items()):
            log = FIXTURES / build / f"{case_id}.log"
            if f"{case_id}.raw" not in entry["outputs"] or entry["stopped"] or not log.is_file():
                continue
            if any(read_step_line(line) for line in read_spice_text(log).splitlines()):
                runs.append((build, case_id))
    return runs


@pytest.mark.parametrize(("build", "case_id"), _stepped_sweeps())
def test_the_raws_steps_are_the_logs_step_lines(build: str, case_id: str, tmp_path: Path):
    parsed = rec.decode(build, case_id, tmp_path)
    rows = parsed.logs.value("steps")
    assert rows == parse_step_iterations(rec.recorded(build, f"{case_id}.log"))
    if parsed.raw.descriptor.step_status == "matched":
        assert [dict(step.parameters) for step in parsed.raw.descriptor.steps] == rows


def test_the_corpus_has_stepped_sweeps_on_both_builds():
    assert {build for build, _ in _stepped_sweeps()} == set(rec.BUILDS)


def _step_patterns() -> dict[str, list[str]]:
    """Every string constant of the package that is a regex reading ``.step``."""
    found: dict[str, list[str]] = {}
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.search(r"\\\.step", node.value, re.I)
            ):
                found.setdefault(path.relative_to(SOURCE).as_posix(), []).append(node.value)
    return found


def test_only_the_log_parser_reads_a_step_line():
    assert list(_step_patterns()) == ["lib/log_parser.py"]
