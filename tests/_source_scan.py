"""The package's source, parsed once per process, for the tests that hold a
rule to the one place that owns it."""

from __future__ import annotations

import ast
import functools
from pathlib import Path

#: The package's own source folder.
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"


@functools.cache
def parsed(path: Path) -> ast.Module:
    """The module at ``path``, parsed; each file is read once per process."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def modules() -> dict[str, ast.Module]:
    """Every module of the package, parsed, by its path under the package
    (``lib/sheet.py``)."""
    return {
        path.relative_to(PACKAGE).as_posix(): parsed(path)
        for path in sorted(PACKAGE.rglob("*.py"))
    }
