"""The names of a log's sections and of a capture's files are written once.

``log_types.LOG_SECTIONS`` lists the sections the decoder writes and the
validated view checks; ``parser_capture.CAPTURED_NAMES`` names the files a
capture holds, which every decoder of a capture verifies.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ltspice_mcp.lib.decoded_log import DecodedLog, LogSectionName
from ltspice_mcp.lib.log_decode import decode_logs
from ltspice_mcp.lib.log_types import LOG_SECTIONS
from ltspice_mcp.lib.parser_capture import CAPTURED_NAMES
from tests.test_log_decode import LIMITS, capture

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"
FIXTURES = Path(__file__).parent / "fixtures"


def test_the_decoder_writes_the_sections_the_view_reads(tmp_path: Path):
    captured, directory = capture(tmp_path, source=FIXTURES / "ltspice_tran_rc.log")
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert [key for key in facts if key not in ("version", "capture_facts", "scan")] == list(
        LOG_SECTIONS
    )
    logs = DecodedLog(facts)
    names: tuple[LogSectionName, ...] = LOG_SECTIONS
    assert logs.as_dict() == facts
    assert {name: logs.section(name) for name in names} == {name: facts[name] for name in names}
    assert {item.role: item.name for item in captured.files} == {
        role: CAPTURED_NAMES[role] for role in ("log",)
    }


def _modules_with(predicate) -> list[str]:
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(predicate(node) for node in ast.walk(tree)):
            found.append(path.relative_to(SOURCE).as_posix())
    return found


def _lists_every_section(node: ast.AST) -> bool:
    elements = getattr(node, "elts", None)
    if not isinstance(node, (ast.Tuple, ast.List)) or elements is None:
        return False
    values = [item.value for item in elements if isinstance(item, ast.Constant)]
    return set(values) == set(LOG_SECTIONS)


def test_the_section_names_are_listed_once():
    assert _modules_with(_lists_every_section) == ["lib/log_types.py"]


def test_the_captured_file_names_are_written_once():
    def names_a_capture_file(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value == CAPTURED_NAMES["console"]

    assert _modules_with(names_a_capture_file) == ["lib/parser_capture.py"]
