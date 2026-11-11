"""The job status vocabulary: one list, split once into live and terminal.

A status a set left out reads as still running in one place and as finished in
another: a wait that returns at once while its receipt says to wait again.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import get_args

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_types import AnalysisStatus, ExperimentCaseStatus, ExperimentStatus
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES, TERMINAL_STATUSES
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import handle_jobs
from tests._source_scan import PACKAGE, parsed, scopes
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment

#: The modules that own the vocabulary and the transitions between its words.
_OWNERS = frozenset({"lib/experiment_types.py", "lib/job_types.py", "lib/job_lifecycle.py"})
#: Every word a job record's status may be.
_JOB_STATUS_WORDS = frozenset(get_args(ExperimentStatus))
#: The job status words no case or analysis status shares: one of them alone
#: says a collection holds a job's statuses.
_JOB_ONLY_WORDS = (
    _JOB_STATUS_WORDS
    - frozenset(get_args(ExperimentCaseStatus))
    - frozenset(get_args(AnalysisStatus))
)


def test_live_and_terminal_statuses_split_the_vocabulary() -> None:
    vocabulary = set(get_args(ExperimentStatus))
    assert vocabulary == TERMINAL_STATUSES | NON_TERMINAL_LIVE_STATUSES
    assert not TERMINAL_STATUSES & NON_TERMINAL_LIVE_STATUSES


_COLLECTIONS = (ast.Set, ast.Tuple, ast.List)


def _collection_strings(node: ast.AST) -> set[str]:
    """The strings of a set, tuple or list display, or of ``frozenset``,
    ``set``, ``tuple`` or ``list`` over one; empty for any other node."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple", "list"}
        and node.args
    ):
        node = node.args[0]
    if not isinstance(node, _COLLECTIONS):
        return set()
    return {
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def _holds_job_statuses(strings: set[str]) -> bool:
    """Two or more job status words, or one that only a job's status has."""
    return len(strings & _JOB_STATUS_WORDS) >= 2 or bool(strings & _JOB_ONLY_WORDS)


def _status_sets_in(tree: ast.AST) -> set[str]:
    """The scopes in ``tree`` that write a collection of job status words
    (``_holds_job_statuses``), a ``Literal[...]`` of them included."""
    return {
        name
        for name, nodes in scopes(tree)
        if any(_holds_job_statuses(_collection_strings(node)) for node in nodes)
    }


#: The scopes outside the owners that write two or more job status words, each
#: with what its collection holds. A case's and an attached analysis's
#: statuses share words with a job's ("failed", "cancelled"), so a set of
#: theirs is listed here too.
_STATUS_SETS: dict[str, str] = {
    "lib/experiment_resume.py:_make_child": "case statuses: a case that ended with no output",
    "lib/experiment_runner.py:ExperimentRunner.materialize_job": (
        "case statuses: a case that ended with no output"
    ),
    "lib/experiment_runner.py:ExperimentRunner._mark_case": (
        "case statuses: the type of a terminal case status, and a case that ended with no output"
    ),
    "lib/experiment_runner.py:ExperimentRunner._finish_unstarted_analysis": (
        "analysis statuses: the type of an analysis that never started"
    ),
    "lib/native_execution.py:prepare_native_cases": (
        "case statuses: a case that ended with no output"
    ),
    "lib/observability.py:<module>": "job event names, a vocabulary of their own (JobEvent)",
    "tools/receipts.py:<module>": "the completeness counters a receipt's schema requires",
    "tools/receipts.py:_recovery_routes": "analysis statuses: an analysis that failed",
}


def test_no_other_module_spells_a_set_of_job_statuses() -> None:
    """Every other module reads ``TERMINAL_STATUSES`` and
    ``NON_TERMINAL_LIVE_STATUSES`` rather than writing its own: no set, tuple,
    list or ``Literal`` of two or more job status words, in any scope."""
    found = {
        f"{path.relative_to(PACKAGE).as_posix()}:{name}"
        for path in PACKAGE.rglob("*.py")
        if path.relative_to(PACKAGE).as_posix() not in _OWNERS
        for name in _status_sets_in(parsed(path))
    }
    assert found == set(_STATUS_SETS)


def test_the_status_set_rule_catches_what_it_names() -> None:
    tree = ast.parse(
        "def failure(job):\n"
        "    return job.status in {'failed', 'timeout', 'interrupted'}\n"
        "def finished(job):\n"
        "    return job.status in ('completed', 'cancelled')\n"
        "def listed(job):\n"
        "    return job.status not in ['queued', 'running']\n"
        "FROZEN = frozenset({'completed_with_failures', 'failed'})\n"
        "class Job:\n"
        "    LIVE = frozenset(['queued', 'analyzing'])\n"
        "    async def wait(self):\n"
        "        return tuple(('running', 'interrupted'))\n"
        "def typed(status: Literal['completed', 'failed']):\n"
        "    def inner():\n"
        "        return set(('queued', 'cancelled'))\n"
        "    return inner\n"
        "def single(job):\n"
        "    return job.status in {'failed', 'skipped'}, job.status == 'completed'\n"
        "def read(job):\n"
        "    return job.status in TERMINAL_STATUSES, {'produced', 'skipped'}\n"
        "def mixed(job):\n"
        "    return job.status in {'interrupted', 'skipped'}\n"
    )
    assert _status_sets_in(tree) == {
        "failure",
        "finished",
        "listed",
        "<module>",
        "Job",
        "Job.wait",
        "typed",
        "typed.inner",
        "mixed",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", ["timeout", "paused"])
async def test_a_record_with_a_status_outside_the_vocabulary_is_refused(
    unknown: str, state_no_sim: SessionState, work_dir: Path
):
    """A hand-edited or foreign record whose status this build has no word for
    is refused like any other malformed record, rather than read as live by one
    reader and finished by another."""
    job = _experiment(work_dir, _circuit(work_dir), status="completed", case_status="produced")
    experiment_store.save_job(job)
    record = json.loads(job.store_path.read_text(encoding="utf-8"))
    record["status"] = unknown
    job.store_path.write_text(json.dumps(record), encoding="utf-8")

    assert experiment_store.load_job(job.job_id, work_dir) is None
    for action in ("status", "wait", "runs"):
        data = _assert_jobs_schema(
            await handle_jobs(_args(action, job_id=job.job_id), state_no_sim)
        )
        assert data["error"]["code"] == "job_not_found", (action, data)
