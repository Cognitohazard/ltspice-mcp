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
import itertools
from dataclasses import dataclass

from tests._source_scan import modules, scopes


@dataclass(frozen=True)
class _LoopWait:
    reason: str
    #: The work that ends in this wait, and the function that runs it on
    #: threads of its own, when the wait is allowed because of where it runs.
    work: str | None = None
    runner: str | None = None
    #: The functions between the wait and the work, nearest the wait first:
    #: the wait's own function is used only inside the first, each one only
    #: inside the next, and the last only inside the work.
    between: tuple[str, ...] = ()


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
        between=("_show_from_sheet",),
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


def _inside(tree: ast.AST, function: str) -> set[int]:
    """The ids of every node inside a function named ``function``."""
    return {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
        for inner in ast.walk(node)
    }


def test_each_wait_is_reached_only_from_its_work():
    """The function that waits is reached only from the listed work, so no new
    caller (an ``asyncio.to_thread`` of it, or another hook that reaches it)
    can wait from a thread of the default pool."""
    for site, wait in _LOOP_WAITS.items():
        if wait.work is None:
            continue
        module, scope = site.split(":")
        chain = (scope.rsplit(".", 1)[-1], *wait.between, wait.work)
        for used, owner in itertools.pairwise(chain):
            for path, tree in modules().items():
                # A public name such as ``place`` is taken elsewhere for other
                # things; a private one is this module's own wherever it is used.
                if path != module and not used.startswith("_"):
                    continue
                inside = _inside(tree, owner)
                for node in _uses(tree, used):
                    assert id(node) in inside, (
                        f"{path}:{getattr(node, 'lineno', '?')} uses {used} outside "
                        f"{owner}, which keeps {site} off the default pool"
                    )


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


# ---------------------------------------------------------------------------
# The cancel gate
# ---------------------------------------------------------------------------

#: Every place that holds a job's cancellation lock, with every call made while
#: it is held. The lock is the gate between a launch and a cancel from another
#: process, so whatever runs under it holds every launch of the job and every
#: cancel back: nothing under it may wait on the event loop or on a write of
#: the job's record. A new call under the lock fails here until it is listed.
_CANCEL_GATE: dict[str, tuple[frozenset[str], str]] = {
    "lib/experiment_runner.py:ExperimentRunner._submit_case_under_cancel_gate": (
        frozenset({"cancellation_requested", "is_set", "now"}),
        "A launch's decision: no cancel marker and no stop, then its submitted "
        "stamp. Its launch, launch record and simulator start come after.",
    ),
    "lib/experiment_store.py:request_cancellation": (
        frozenset(
            {
                "PermissionError",
                "atomic_write_json",
                "cancellation",
                "control_authorized",
                "ensure_root",
                "envelope",
                "isoformat",
                "load_job",
                "now",
            }
        ),
        "A cancel from another process: it reads the record to authorize the "
        "cancel and writes the marker, the gate's own small record; it waits on "
        "no launch and no loop.",
    ),
    "lib/experiment_store.py:_delete_unretained_job": (
        frozenset({"cancellation", "suppress", "unlink"}),
        "Removing a finished job's record and marker.",
    ),
}

_LOCKS = frozenset({"file_lock", "async_file_lock"})


def _cancel_gates() -> dict[str, tuple[set[str], bool]]:
    """Each scope holding a cancellation lock: the calls made under it, and
    whether anything under it is awaited."""
    found: dict[str, tuple[set[str], bool]] = {}
    for module, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if not isinstance(node, (ast.With, ast.AsyncWith)):
                    continue
                if not any(
                    isinstance(item.context_expr, ast.Call)
                    and _name(item.context_expr.func) in _LOCKS
                    and any(
                        isinstance(inner, ast.Call) and _name(inner.func) == "cancellation_lock"
                        for inner in ast.walk(item.context_expr)
                    )
                    for item in node.items
                ):
                    continue
                inside = [inner for statement in node.body for inner in ast.walk(statement)]
                calls = {
                    name
                    for inner in inside
                    if isinstance(inner, ast.Call) and (name := _name(inner.func)) is not None
                }
                awaited = any(isinstance(inner, ast.Await) for inner in inside)
                site = f"{module}:{scope}"
                previous, previously_awaited = found.get(site, (set(), False))
                found[site] = (previous | calls, previously_awaited or awaited)
    return found


def test_the_cancel_gate_holds_no_wait_on_the_loop_or_a_write():
    """Under a job's cancellation lock run only the listed calls, and nothing
    is awaited, so the gate never waits for a launch record, a write of the
    job's record or the loop."""
    found = _cancel_gates()
    assert set(found) == set(_CANCEL_GATE), (
        "places that hold a job's cancellation lock changed; list each with its calls "
        "and why none of them waits on a write or the loop"
    )
    for site, (calls, awaited) in found.items():
        allowed, _reason = _CANCEL_GATE[site]
        assert not awaited, f"{site} awaits under the cancellation lock"
        assert calls == allowed, (
            f"{site} calls {sorted(calls - allowed)} under the cancellation lock and no "
            f"longer calls {sorted(allowed - calls)}; a launch or a cancel waits on whatever "
            "runs there"
        )


# ---------------------------------------------------------------------------
# The order of a job's writes
# ---------------------------------------------------------------------------

#: Every place that uses ``experiment_store.save_job``. A job the registry
#: keeps is written through its writes, one at a time per job in the order
#: asked for (``JobRegistry._persist_locks``), where an older write cannot land
#: on top of a newer one; a write outside them could. Each other use writes a record no ordered write of the job can be in
#: flight for, and says why. A new use fails here until it is listed.
_SAVE_JOB_USES: dict[str, str] = {
    "lib/job_registry.py:JobRegistry._persist_async.persist": (
        "The registry's strict write, under the job's write lock."
    ),
    "lib/job_registry.py:JobRegistry._persist_sync": (
        "The registry's write: under the job's write lock on the loop "
        "(``_persist_async``), or in line where there is no loop."
    ),
    "lib/experiment_resume.py:persist_reconciled": (
        "Called only by the registry's write, under the job's write lock, "
        "which it re-reads and saves inside."
    ),
    "lib/experiment_resume.py:_persist_discovery": (
        "A recoverable job's first record under its new owner, at admission, "
        "adoption or a resume's child, written before its coordinator starts: "
        "no ordered write of it has been asked for yet."
    ),
    "lib/experiment_runner.py:ExperimentRunner._claim_request_id": (
        "A job's first record, written at admission inside the request gate "
        "before its coordinator starts: no ordered write of it exists yet."
    ),
    "lib/experiment_runner.py:ExperimentRunner._record_index_note.save": (
        "Only with persistence off, when the registry writes nothing."
    ),
}


def test_every_write_of_a_job_record_outside_the_registry_is_listed():
    """A job's record is written by the registry's ordered writes, or by a
    listed writer that no ordered write of the job can race."""
    found = {
        f"{module}:{scope}"
        for module, tree in modules().items()
        for scope, nodes in scopes(tree)
        if any(
            isinstance(node, (ast.Name, ast.Attribute)) and _name(node) == "save_job"
            for node in nodes
        )
        and not (module == "lib/experiment_store.py" and scope == "save_job")
    }
    assert found - set(_SAVE_JOB_USES) == set(), (
        "a job record is written outside the registry's ordered writes: write it "
        "through JobRegistry.persist_job or persist_strict, or list why no ordered "
        "write of the job can be in flight"
    )
    assert set(_SAVE_JOB_USES) - found == set(), "listed writers that no longer exist"
