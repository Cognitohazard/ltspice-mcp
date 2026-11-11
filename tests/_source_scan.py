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


def own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope``'s body, not those of a function or class defined
    in it, which is a scope of its own."""
    own: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        own.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return own


def scopes(tree: ast.AST, prefix: str = "") -> list[tuple[str, list[ast.AST]]]:
    """Each scope of ``tree`` with the nodes of its own body: a module's top
    level as ``<module>``, then every class body and every function, ``async
    def`` included, qualified by the ones around it (``Ops.outer.inner``)."""
    found: list[tuple[str, list[ast.AST]]] = []
    if isinstance(tree, ast.Module):
        found.append(("<module>", own_nodes(tree)))
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}{node.name}"
            found.append((name, own_nodes(node)))
            found.extend(scopes(node, f"{name}."))
        else:
            found.extend(scopes(node, prefix))
    return found
