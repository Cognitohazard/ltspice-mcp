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
from ltspice_mcp.lib.experiment_types import ExperimentStatus
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES, TERMINAL_STATUSES
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import receipts
from ltspice_mcp.tools.jobs import handle_jobs
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment

_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"
#: The modules that own the vocabulary and the transitions between its words.
_OWNERS = frozenset({"lib/experiment_types.py", "lib/job_types.py", "lib/job_lifecycle.py"})
#: Words only a set of job statuses holds: no case or analysis status is
#: ``completed_with_failures``, and ``queued`` is a job's status and a case's.
_JOB_STATUS_WORDS = frozenset({"completed_with_failures", "queued"})


def test_live_and_terminal_statuses_split_the_vocabulary() -> None:
    vocabulary = set(get_args(ExperimentStatus))
    assert vocabulary == TERMINAL_STATUSES | NON_TERMINAL_LIVE_STATUSES
    assert not TERMINAL_STATUSES & NON_TERMINAL_LIVE_STATUSES
    # What a receipt calls finished is the record's own terminal set.
    assert receipts.TERMINAL_EXPERIMENT_STATUSES == TERMINAL_STATUSES


def _set_literal_strings(node: ast.AST) -> set[str] | None:
    """The strings of a set display, or of ``frozenset``/``set`` over a list or
    tuple display (one over a set display is the set display)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set"}
        and node.args
        and isinstance(node.args[0], ast.List | ast.Tuple)
    ):
        node = node.args[0]
    elif not isinstance(node, ast.Set):
        return None
    assert isinstance(node, ast.Set | ast.List | ast.Tuple)
    return {
        elt.value for elt in node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def test_no_other_module_spells_a_set_of_job_statuses() -> None:
    """Every other module reads ``TERMINAL_STATUSES`` and
    ``NON_TERMINAL_LIVE_STATUSES`` rather than writing its own."""
    found = sorted(
        f"{path.relative_to(_SRC).as_posix()}:{getattr(node, 'lineno', 0)}"
        for path in _SRC.rglob("*.py")
        if path.relative_to(_SRC).as_posix() not in _OWNERS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if (strings := _set_literal_strings(node)) and strings & _JOB_STATUS_WORDS
    )
    assert not found, f"job status sets spelled outside {sorted(_OWNERS)}: {found}"


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
        data = _assert_jobs_schema(await handle_jobs(_args(action, job_id=job.job_id), state_no_sim))
        assert data["error"]["code"] == "job_not_found", (action, data)
