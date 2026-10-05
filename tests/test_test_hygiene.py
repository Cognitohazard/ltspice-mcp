"""Waiting rules the suite holds itself to, checked rather than documented.

Every race the suite has hit had the same shape: a test waited on something
other than what it then asserted — a file disappearing, a fixed sleep, a
timeout sized for the machine it was written on — and lost on a slower
runner. ``docs/TESTING.md`` said so for weeks before the next one was
written. These checks make the rules mechanical.

Source side: every task the server starts has an owner that can await it.
A direct ``create_task``/``ensure_future`` outside ``BackgroundTasks`` must be
listed here with that owner, so a new one is a decision someone wrote down.

Test side: see the rules below ``_SOURCE_SPAWNS``. A line that has to break
one says why with a ``# timing: <reason>`` comment on it.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

_SPAWNERS = frozenset({"create_task", "ensure_future"})

_SOURCE_SPAWNS: dict[str, tuple[int, str]] = {
    "ltspice_mcp/lib/background.py:BackgroundTasks.spawn": (
        1,
        "the owner primitive itself",
    ),
    "ltspice_mcp/lib/experiment_runner.py:ExperimentRunner.start_committed": (
        1,
        "the job's coordinator, kept on job.task; wait() and settled() await it",
    ),
    "ltspice_mcp/lib/experiment_runner.py:ExperimentRunner._run_job": (
        3,
        "case tasks are gathered; the deadline and external-cancel watchers are "
        "cancelled and awaited before the coordinator returns",
    ),
    "ltspice_mcp/lib/experiment_runner.py:ExperimentRunner._await_case": (
        1,
        "the cancel wait is awaited, then cancelled, in the same function",
    ),
    "ltspice_mcp/lib/experiment_runner.py:ExperimentRunner._run_analysis": (
        2,
        "the analysis and the cancel wait are awaited in the same function",
    ),
    "ltspice_mcp/lib/job_registry.py:_issue_cancels": (
        1,
        "awaited under the shutdown bound; stragglers are cancelled",
    ),
    "ltspice_mcp/lib/services.py:bounded_parse": (
        1,
        "awaited under its deadline; a worker thread past it cannot be stopped",
    ),
    "ltspice_mcp/tools/run_code.py:CodeWorker._read_message": (
        1,
        "the line read is awaited in the same function",
    ),
    "ltspice_mcp/tools/run_code.py:CodeWorker.run": (
        1,
        "kept as the worker's drain, which its next call and close await or cancel",
    ),
    "ltspice_mcp/tools/verify.py:evaluate_verify_circuit": (
        1,
        "the render is awaited before the reply is built",
    ),
}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class _Calls(ast.NodeVisitor):
    """Every call in a module, with the qualified name of the scope it is in."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.calls: list[tuple[str, ast.Call]] = []

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((".".join(self.scope), node))
        self.generic_visit(node)


def _calls(path: Path) -> list[tuple[str, ast.Call]]:
    visitor = _Calls()
    visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
    return visitor.calls


def _source_spawns() -> Counter[str]:
    found: Counter[str] = Counter()
    for path in sorted((SRC / "ltspice_mcp").rglob("*.py")):
        module = path.relative_to(SRC).as_posix()
        for scope, call in _calls(path):
            if _call_name(call) in _SPAWNERS:
                found[f"{module}:{scope}"] += 1
    return found


def test_every_task_the_server_starts_has_an_owner():
    """A new direct spawn fails here until its owner is written down above.

    Prefer ``BackgroundTasks.spawn`` on the owner (the runner, the registry,
    the session): then ``settled`` waits for it and nothing needs listing.
    """
    found = _source_spawns()
    expected = {site: count for site, (count, _owner) in _SOURCE_SPAWNS.items()}
    unlisted = {site: n for site, n in found.items() if expected.get(site) != n}
    stale = {site: n for site, n in expected.items() if site not in found}
    assert not unlisted, (
        "tasks started without a listed owner (site: count found); spawn them "
        f"through the owner's BackgroundTasks, or list the owner here: {unlisted}"
    )
    assert not stale, f"listed spawn sites that no longer exist: {stale}"
