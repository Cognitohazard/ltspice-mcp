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
import functools
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


# ---------------------------------------------------------------------------
# Test side
# ---------------------------------------------------------------------------

TESTS = ROOT / "tests"
_PRAGMA = "# timing:"
_LIVENESS_S = 30.0
"""``tests/conftest.py``'s ``LIVENESS_S``, restated so this file reads no
fixtures; a test below keeps the two equal."""

_SLEEPS = frozenset({"sleep"})
_WAITS = frozenset(
    {
        "wait",
        "wait_for",
        "join",
        "result",
        "communicate",
        "await_until",
        "wait_until",
        "_jobs_wait",
    }
)
_POLLS = frozenset({"await_until", "wait_until"})
_OFFLOADS = frozenset({"to_thread"})
"""Calls that run a function handed to them, so ``to_thread(event.wait, 5)``
caps a wait as surely as ``event.wait(5)`` does."""
_TIMEOUT_KEYWORDS = frozenset({"timeout", "timeout_s"})
_DWELL_KEYWORDS = frozenset({"wait_s"})
_BOUND_KEYWORDS = _TIMEOUT_KEYWORDS | _DWELL_KEYWORDS
_FILE_PROBES = frozenset({"exists", "is_file", "is_dir"})


def _test_files() -> list[Path]:
    skip = {"test_test_hygiene.py", "schedule_jitter.py"}
    return sorted(path for path in TESTS.glob("*.py") if path.name not in skip)


def _comments(source: str) -> dict[int, str]:
    import io
    import tokenize

    found: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            found[token.start[0]] = token.string
    return found


_COMPOUND = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)


def _excused(node: ast.AST, statement: ast.stmt | None, comments: dict[int, str]) -> bool:
    """A ``# timing: <reason>`` comment on the lines of the node or of the
    simple statement holding it, or in the comment block directly above."""
    spans = [node]
    if statement is not None and not isinstance(statement, _COMPOUND):
        spans.append(statement)
    first = min(getattr(span, "lineno", 0) for span in spans)
    last = max(getattr(span, "end_lineno", None) or getattr(span, "lineno", 0) for span in spans)
    lines = list(range(first, last + 1))
    above = first - 1
    while above in comments:
        lines.append(above)
        above -= 1
    for line in lines:
        comment = comments.get(line, "")
        if comment.startswith(_PRAGMA) and comment[len(_PRAGMA) :].strip():
            return True
    return False


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _clock_offset(node: ast.AST) -> float | None:
    """``N`` in ``time.monotonic() + N``: a deadline handed to the code under test."""
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and isinstance(node.left, ast.Call)
        and _call_name(node.left) == "monotonic"
    ):
        return _number(node.right)
    return None


def _callable_name(node: ast.AST) -> str | None:
    """The name of a function passed by reference: ``wait`` in ``event.wait``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _short(value: float | None) -> bool:
    return value is not None and 0 < value < _LIVENESS_S


def _only_probes_files(predicate: ast.AST) -> bool:
    """Whether a poll's predicate asks only whether a path exists.

    Such a wait ends when a file appears or goes, which is rarely the thing the
    test then reads: a file another process writes can exist before its
    content lands, and a file removed by background work goes before that
    work has finished. Return the parsed content, or await the work's owner.
    """
    if isinstance(predicate, ast.Attribute):
        return predicate.attr in _FILE_PROBES
    probes = {_call_name(inner) for inner in ast.walk(predicate) if isinstance(inner, ast.Call)}
    return bool(probes & _FILE_PROBES) and not probes - _FILE_PROBES - {"bool", "all", "any"}


class _TestScan(ast.NodeVisitor):
    def __init__(self, path: Path, comments: dict[int, str]) -> None:
        self.path = path
        self.comments = comments
        self.scope: list[str] = []
        self.statements: list[ast.stmt] = []
        self.found: dict[str, list[str]] = {"sleep": [], "file-poll": [], "short-wait": []}

    def visit(self, node: ast.AST) -> None:
        if not isinstance(node, ast.stmt):
            super().visit(node)
            return
        self.statements.append(node)
        try:
            super().visit(node)
        finally:
            self.statements.pop()

    def _flag(self, rule: str, node: ast.AST, what: str) -> None:
        statement = self.statements[-1] if self.statements else None
        if not _excused(node, statement, self.comments):
            where = f"{self.path.name}:{getattr(node, 'lineno', '?')}"
            self.found[rule].append(f"{where} {'.'.join(self.scope)}: {what}")

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_defaults(node)
        self._scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def _check_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        params = [*node.args.args, *node.args.kwonlyargs]
        defaults = [None] * (len(node.args.args) - len(node.args.defaults))
        defaults += [*node.args.defaults, *node.args.kw_defaults]
        for param, default in zip(params, defaults, strict=True):
            if param.arg in _DWELL_KEYWORDS | _TIMEOUT_KEYWORDS and _short(_number(default)):
                self._flag(
                    "short-wait", default or node, f"default {param.arg}={_number(default):g}"
                )

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value in _DWELL_KEYWORDS
                and _short(_number(value))
            ):
                self._flag("short-wait", value, f'"{key.value}": {_number(value):g}')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name in _SLEEPS and node.args:
            value = _number(node.args[0])
            if value != 0.0 and self.scope[-1:] not in (["wait_until"], ["await_until"]):
                self._flag("sleep", node, f"{name}({ast.unparse(node.args[0])})")
        if name in _POLLS and node.args and _only_probes_files(node.args[0]):
            self._flag("file-poll", node, ast.unparse(node.args[0])[:80])
        if name in _WAITS:
            for arg in node.args:
                if _short(_number(arg)):
                    self._flag("short-wait", node, f"{name}(..., {_number(arg):g})")
        if name in _OFFLOADS and node.args and _callable_name(node.args[0]) in _WAITS:
            for arg in node.args[1:]:
                if _short(_number(arg)):
                    waited = _callable_name(node.args[0])
                    self._flag("short-wait", node, f"{name}({waited}, {_number(arg):g})")
        for keyword in node.keywords:
            if keyword.arg in _BOUND_KEYWORDS and _short(_number(keyword.value)):
                self._flag("short-wait", node, f"{keyword.arg}={_number(keyword.value):g}")
            if keyword.arg == "deadline" and _short(_clock_offset(keyword.value)):
                self._flag("short-wait", node, f"deadline=now+{_clock_offset(keyword.value):g}")
        self.generic_visit(node)


@functools.cache
def _scan_tests() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"sleep": [], "file-poll": [], "short-wait": []}
    for path in _test_files():
        source = path.read_text(encoding="utf-8")
        scan = _TestScan(path, _comments(source))
        scan.visit(ast.parse(source))
        for rule, rows in scan.found.items():
            found[rule].extend(rows)
    return found


def _scan_source(source: str) -> dict[str, list[str]]:
    scan = _TestScan(TESTS / "sample.py", _comments(source))
    scan.visit(ast.parse(source))
    return scan.found


def test_the_liveness_cap_restated_here_is_conftests():
    from tests.conftest import LIVENESS_S

    assert _LIVENESS_S == LIVENESS_S


def test_no_test_sleeps_in_place_of_waiting():
    """A fixed sleep is a guess at how long something takes. Wait on the
    event or state the test then reads; a sleep that simulates work, or a
    window showing something does not happen, says so with a reason."""
    found = _scan_tests()["sleep"]
    assert not found, "sleeps without a '# timing: <reason>':\n" + "\n".join(found)


def test_no_test_polls_for_a_file_to_exist_or_go():
    """A file exists before its writer's content lands, and a file background
    work removes is gone before that work has finished. Poll for the parsed
    content (``written``), or await the work's owner (``settled``)."""
    found = _scan_tests()["file-poll"]
    assert not found, "polls on a path's existence:\n" + "\n".join(found)


def test_every_wait_is_capped_at_the_liveness_bound():
    """A wait that gives up before ``LIVENESS_S`` claims the runner is fast.
    Waits that test a timeout, and a dwell carried as data, say so."""
    found = _scan_tests()["short-wait"]
    assert not found, (
        "waits capped below LIVENESS_S without a '# timing: <reason>':\n" + "\n".join(found)
    )


def test_no_test_names_a_process_by_its_pid_alone():
    """Windows hands a freed pid to the next process quickly, and parallel test
    workers start processes all the time, so ``psutil.pid_exists`` on a reaped
    worker's pid can find a stranger. Ask ``process_running``
    (``tests/conftest.py``), which also matches a process by its start time."""
    found = [
        f"{path.relative_to(ROOT)}:{call.lineno}"
        for path in _test_files()
        if path.name != "conftest.py"
        for _scope, call in _calls(path)
        if _call_name(call) == "pid_exists"
    ]
    assert not found, "pid checks that ignore reuse:\n" + "\n".join(found)


def test_a_process_is_told_from_a_later_one_on_its_pid():
    """``process_running`` matches by start time: the same pid started at
    another moment is another process, and a process this test started is
    checked against the start recorded when it was spawned."""
    import subprocess
    import sys

    import psutil

    from tests.conftest import LIVENESS_S, identify, process_running

    me = psutil.Process()
    assert process_running(identify(me.pid))
    assert process_running(me.pid, me.create_time())
    assert not process_running(me.pid, me.create_time() - 1)

    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        assert process_running(child.pid)
    finally:
        child.communicate(timeout=LIVENESS_S)
    assert not process_running(child.pid)


def test_a_process_that_has_exited_is_not_running():
    """An exit is an exit, whatever psutil can still find.

    psutil reads a Windows process as running when its exit code is 259 or its
    pid is still listed, and a process object outlives its process while this
    test's Popen holds a handle to it. A parser worker the job had confirmed
    gone read as running that way on Windows CI. Exiting with 259 makes it
    certain rather than a matter of timing.
    """
    import subprocess
    import sys

    from tests.conftest import LIVENESS_S, process_running

    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(259)"])
    child.wait(timeout=LIVENESS_S)
    assert not process_running(child.pid)


def test_the_rules_catch_what_they_name():
    """Each rule flags its pattern and a reason excuses it, on the line, on
    the statement, or in the comment block above it."""
    flagged = _scan_source(
        "async def test_x(path, runner, job):\n"
        "    await asyncio.sleep(0.5)\n"
        "    await await_until(lambda: not path.exists())\n"
        "    wait_until(path.is_file)\n"
        "    await runner.wait(job, 1)\n"
        "    payload = {'execution': {'wait_s': 5}}\n"
        "    proc.wait(timeout=10)\n"
        "    run(deadline=time.monotonic() + 5)\n"
        "    await asyncio.to_thread(entered.wait, 5)\n"
    )
    assert len(flagged["sleep"]) == 1
    assert len(flagged["file-poll"]) == 2
    assert len(flagged["short-wait"]) == 5

    excused = _scan_source(
        "async def test_x(runner, job):\n"
        "    await asyncio.sleep(0)\n"
        "    await asyncio.sleep(0.5)  # timing: fake work\n"
        "    # timing: asserts the wait times out\n"
        "    assert not await runner.wait(\n"
        "        job, 0.01\n"
        "    )\n"
        "    await runner.wait(job, LIVENESS_S)\n"
        "    proc.wait(timeout=60)\n"
        "    await asyncio.to_thread(entered.wait, LIVENESS_S)\n"
    )
    assert excused == {"sleep": [], "file-poll": [], "short-wait": []}

    bare = _scan_source("async def test_x():\n    await asyncio.sleep(1)  # timing:\n")
    assert len(bare["sleep"]) == 1, "a reason has to say something"
