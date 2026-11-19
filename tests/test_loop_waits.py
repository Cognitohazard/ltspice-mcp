"""No thread waits on the event loop from a pool the loop needs to answer it.

A thread that hands a coroutine to the loop (``run_coroutine_threadsafe``) and
waits for it must not be a thread of the loop's default pool: the loop answers
it with work that runs on that pool (every file write and parse it offloads),
and once the waiting threads fill the pool that work can never run, so every
one of them waits for ever and so does every other offloaded read.

Every place that hands a coroutine to a loop from a thread is listed below with
the thread that waits for it and why the loop needs no thread of that thread's
pool. Where the reason is that the waiting work runs on threads of its own, the
entry names the work and the function that runs it there, and the check holds
every use of the work to that function, so a second way onto the default pool
cannot come back.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tests._source_scan import modules, scopes


@dataclass(frozen=True)
class _LoopWait:
    reason: str
    #: The work that ends in this wait, and the function that runs it on
    #: threads of its own, when the wait is allowed because of where it runs.
    work: str | None = None
    runner: str | None = None


_LOOP_WAITS: dict[str, _LoopWait] = {
    "lib/experiment_runner.py:ExperimentRunner._record_launch_intent": _LoopWait(
        "A recoverable case's launch waits for its launch record. It runs on "
        "``_LAUNCH_THREADS``; the record is a write on the default pool, and a "
        "stop of the job ends the wait (``_record_launch``).",
        work="_submit_case_under_cancel_gate",
        runner="_in_launch_thread",
    ),
    "tools/analysis.py:_JobRun.place": _LoopWait(
        "Placing a job's results beside a sheet for an open LTspice window. It "
        "runs on ``_WINDOW_THREADS``; the placement's file work runs on the "
        "default pool.",
        work="_show_in_ltspice",
        runner="_in_window_thread",
    ),
    "api/_session.py:Api._submit_to_loop": _LoopWait(
        "The Python API's private loop runs on a thread of its own; its callers "
        "wait from their own threads, never from a thread of that loop's pool, "
        "and the loop thread itself is refused (``_check_process_and_thread``)."
    ),
}

_DEFAULT_POOL_OFFLOADS = frozenset({"to_thread"})


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _loop_waits() -> set[str]:
    return {
        f"{module}:{scope}"
        for module, tree in modules().items()
        for scope, nodes in scopes(tree)
        if any(
            isinstance(node, ast.Call) and _name(node.func) == "run_coroutine_threadsafe"
            for node in nodes
        )
    }


def test_every_wait_on_the_loop_from_a_thread_is_listed():
    """A new place that hands a coroutine to the loop from a thread fails
    here until it is listed with the thread that waits and its reason."""
    found = _loop_waits()
    assert found - set(_LOOP_WAITS) == set(), (
        "a thread hands a coroutine to the loop here: run the work that waits on "
        "threads of its own (as ``_in_launch_thread`` does) and list it in _LOOP_WAITS"
    )
    assert set(_LOOP_WAITS) - found == set(), "listed loop waits that no longer exist"


def _uses(tree: ast.AST, work: str) -> list[ast.AST]:
    """Each use of ``work`` by name or attribute, its definition aside."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute)) and _name(node) == work
    ]


def _runner_arguments(tree: ast.AST, runner: str) -> set[int]:
    """The ids of the nodes passed as the work to a call of ``runner``."""
    return {
        id(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _name(node.func) == runner and node.args
    }


def test_work_that_waits_on_the_loop_runs_only_on_its_own_threads():
    """Each listed work is used only as the work its runner runs, so no
    caller can hand it to ``asyncio.to_thread`` or the default pool."""
    for site, wait in _LOOP_WAITS.items():
        if wait.work is None:
            continue
        assert wait.runner is not None
        uses = 0
        for module, tree in modules().items():
            allowed = _runner_arguments(tree, wait.runner)
            for node in _uses(tree, wait.work):
                uses += 1
                assert id(node) in allowed, (
                    f"{module}:{getattr(node, 'lineno', '?')} uses {wait.work} other than "
                    f"as the work of {wait.runner}, which keeps {site} off the default pool"
                )
        assert uses, f"{wait.work} is no longer used; update the entry for {site}"


def test_own_thread_runners_never_use_the_default_pool():
    """A runner named above hands its work to an executor it names, never to
    the default pool (``None``) or ``asyncio.to_thread``."""
    runners = {wait.runner for wait in _LOOP_WAITS.values() if wait.runner is not None}
    seen: set[str] = set()
    for tree in modules().values():
        for node in ast.walk(tree):
            if not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in runners
            ):
                continue
            seen.add(node.name)
            executors = [
                call.args[0]
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and _name(call.func) == "run_in_executor"
                and call.args
            ]
            assert executors, f"{node.name} no longer runs its work on an executor"
            for executor in executors:
                assert not (isinstance(executor, ast.Constant) and executor.value is None), (
                    f"{node.name} runs its work on the default pool"
                )
            assert not any(
                isinstance(call, ast.Call) and _name(call.func) in _DEFAULT_POOL_OFFLOADS
                for call in ast.walk(node)
            ), f"{node.name} runs its work on the default pool"
    assert seen == runners
