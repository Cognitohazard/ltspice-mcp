"""Each question about a sheet is answered in one place, checked rather than documented.

Two readers that each answer one question, such as which points of a sheet are
one net or which directives a text names, drift apart, and then one reply says
two things. These checks walk the server's source and fail where a second
answer starts.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ltspice_mcp"


@functools.cache
def _modules() -> dict[str, ast.Module]:
    """Every module of the package, parsed, by its path under the package."""
    return {
        path.relative_to(PACKAGE).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE.rglob("*.py"))
    }


def _called(name: str) -> list[str]:
    """Where a function or method of that name is called, as ``module:line``."""
    return [
        f"{module}:{node.lineno}"
        for module, tree in _modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            or (isinstance(node.func, ast.Name) and node.func.id == name)
        )
    ]


def test_no_text_is_matched_by_spicelibs_instruction_removers():
    """``remove_directive`` and ``move`` select texts by ``_text_matcher``.
    spicelib's removers match directives only, one by substring and the first
    hit, the other anchored at the start and in any case."""
    assert _called("remove_Xinstruction") == []
    assert _called("remove_instruction") == []
