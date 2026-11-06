r"""A dotted projection path reads one way on every argument that takes one.

``include.fields`` (on ``analyze_results`` and an attached analysis) and
``run_fields`` (on ``run_experiments`` and ``jobs(runs)``) share one resolver,
parameterised by the keys their rows carry: a path starts at one of them or is
refused naming them, a bare name reads under ``value`` where the row nests its
numbers there, an empty segment is refused, ``\.`` is a dot inside a key, a
repeat is read once, and a leaf no row has is said with the keys that are.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ltspice_mcp.lib import result_store
from ltspice_mcp.lib.projection import absent_path_warnings, key_outline
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments, jobs, receipts
from ltspice_mcp.tools.analyze import (
    AnalyzeResultsInput,
    capture_attached_analysis,
    handle_analyze_results,
    render_attached_analysis,
)
from tests._source_scan import PACKAGE, parsed
from tests.conftest import make_experiment_job, stage_recorded_fixture

# ---------------------------------------------------------------------------
# Every dotted-path argument goes through the one resolver
# ---------------------------------------------------------------------------

#: (module, model, field): every input field documented as dotted row paths.
_DOTTED_FIELDS = (
    ("tools/analyze.py", "AnalyzeInclude", "fields"),
    ("tools/experiments.py", "AnalysisInclude", "fields"),
    ("tools/experiments.py", "RunExperimentsInput", "run_fields"),
    ("tools/jobs.py", "JobsRunsInput", "run_fields"),
)

#: The per-argument readings, each of which reads through ``resolve_field_paths``.
_READERS = {
    "tools/analyze.py": "resolve_row_fields",
    "tools/receipts.py": "resolve_run_fields",
}


def _calls(node: ast.AST) -> set[str]:
    return {
        call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", "")
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def _module(path: str) -> ast.Module:
    return parsed(PACKAGE / path)


def test_each_reader_reads_through_the_shared_resolver() -> None:
    for path, name in _READERS.items():
        (function,) = (
            node
            for node in _module(path).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert "resolve_field_paths" in _calls(function), f"{path}:{name}"


@pytest.mark.parametrize(("path", "model", "field"), _DOTTED_FIELDS)
def test_each_dotted_field_is_read_by_the_shared_resolver(
    path: str, model: str, field: str
) -> None:
    (cls,) = (
        node
        for node in ast.walk(_module(path))
        if isinstance(node, ast.ClassDef) and node.name == model
    )
    readers = set(_READERS.values()) | {"resolve_field_paths"}
    validators = [
        function
        for function in cls.body
        if isinstance(function, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "id", "") in {"model_validator", "field_validator"}
            for decorator in function.decorator_list
        )
        and _calls(function) & readers
    ]
    assert validators, f"{model}.{field} is not read by the shared projection resolver"
    mentioned = {
        node.attr if isinstance(node, ast.Attribute) else node.value
        for function in validators
        for decorator_or_body in [function]
        for node in ast.walk(decorator_or_body)
        if isinstance(node, ast.Attribute)
        or (isinstance(node, ast.Constant) and isinstance(node.value, str))
    }
    assert field in mentioned, f"{model}'s resolver call does not read {field!r}"


# ---------------------------------------------------------------------------
# include.fields: analyze_results and the attached analysis agree
# ---------------------------------------------------------------------------

_RECIPE = {"key": "k", "metric": "signal_stats", "signal": "V(out)"}

#: (case, fields, refused): the paths a caller writes, and whether they are refused.
_INCLUDE_CASES = [
    ("bare name", ["rms"], False),
    ("value path", ["value.rms"], False),
    ("identity root", ["source", "value.rms"], False),
    ("escaped dot", [r"value.r\.ms"], False),
    ("unknown root", ["valu.rms"], True),
    ("empty segment", ["value..rms"], True),
    ("absent leaf", ["value.rmss"], False),
    ("repeat", ["rms", "value.rms"], False),
]


async def _analyze_page(state: SessionState, raw: Path, fields: list[str]) -> dict[str, Any]:
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "rc"}],
            "recipes": [_RECIPE],
            "include": {"per_run": True, "fields": fields},
        }
    )
    data = (await handle_analyze_results(args, state)).structured_content
    assert data is not None
    return data


async def _attached_page(state: SessionState, raw: Path, fields: list[str]) -> dict[str, Any]:
    """What run_experiments shows for the same block: the fields as its own
    model read them, over the analysis the job stores (which keeps no view)."""
    block = experiments.AttachedAnalysis.model_validate(
        {"recipes": [_RECIPE], "include": {"per_run": True, "fields": fields}}
    )
    assert block.include is not None
    stored = await capture_attached_analysis(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "rc"}],
                "recipes": [_RECIPE],
                "include": {"per_run": True},
            }
        ),
        state,
    )
    return render_attached_analysis(stored, fields=block.include.fields)


def _refusal(validate: Any) -> str:
    with pytest.raises(ValidationError) as caught:
        validate()
    return str(caught.value)


@pytest.mark.parametrize(
    ("case", "fields", "refused"), _INCLUDE_CASES, ids=[c[0] for c in _INCLUDE_CASES]
)
async def test_include_fields_read_the_same_on_analyze_results_and_the_attached_analysis(
    state_no_sim: SessionState, work_dir: Path, case: str, fields: list[str], refused: bool
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    if refused:
        direct = _refusal(lambda: analyze.AnalyzeInclude.model_validate({"fields": fields}))
        attached = _refusal(lambda: experiments.AnalysisInclude.model_validate({"fields": fields}))
        assert "include.fields path" in direct
        assert (
            direct.split("include.fields path", 1)[1]
            == attached.split("include.fields path", 1)[1]
        )
        return

    page = await _analyze_page(state_no_sim, raw, fields)
    attached = await _attached_page(state_no_sim, raw, fields)
    rows = page["results"]["k"]["per_run"]["items"]
    assert attached["results"]["k"]["per_run"]["items"] == rows
    assert attached["results"]["k"].get("warnings", []) == page["results"]["k"].get("warnings", [])
    if case in {"bare name", "value path", "repeat"}:
        assert rows and all(set(row) == {"value"} and "rms" in row["value"] for row in rows)
    if case == "absent leaf":
        (warning,) = page["results"]["k"].get("warnings", [])
        assert "'value.rmss' is absent" in warning and "rms" in warning


@pytest.mark.parametrize(
    "path",
    [
        "assignments._random_seed",
        "assignments._random_run",
        "assignments.R1",
        "assignments.R2",
        "value.rms",
        "value.rms.deep",
        r"value.v(x1\.out)",
        "value.v(x1.out)",
        "value._private",
        "missing.leaf",
    ],
)
def test_a_saved_key_outline_warns_as_the_rows_it_outlines(path: str) -> None:
    """An attached analysis keeps a key outline in place of its rows and checks a
    projection against it, so the outline must answer every path as the rows
    do. It once left out each key starting with '_', so a projection of a
    random case's 'assignments._random_seed' was called absent from rows that
    all carried it."""
    rows = [
        {
            "source": "a",
            "assignments": {"R1": "1k", "_random_seed": 3, "_random_run": 0},
            "value": {"rms": 1.0, "v(x1.out)": 2.0, "_private": {"x": 1}},
        },
        {"source": "b", "assignments": {"_random_seed": 3, "_random_run": 1}, "value": {}},
    ]

    assert absent_path_warnings([key_outline(rows)], [path], **_WARNING_ARGS) == (
        absent_path_warnings(rows, [path], **_WARNING_ARGS)
    )


_WARNING_ARGS: dict[str, str] = {"argument": "include.fields", "rows_name": "row"}


async def test_a_continued_page_keeps_the_resolved_view(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    """A cursor minted under a bare name pages the same rows as the first page."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    request = {
        "sources": [{"path": str(raw), "label": "rc"}],
        "recipes": [_RECIPE],
        "all_steps": True,
        "include": {"per_run": {"limit": 1}, "fields": ["rms"]},
    }
    first = (
        await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state_no_sim)
    ).structured_content
    assert first is not None
    cursor = first["results"]["k"]["per_run"]["next_cursor"]
    assert cursor is not None
    second = (
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {**request, "include": {"per_run": {"limit": 1, "cursor": cursor}}}
            ),
            state_no_sim,
        )
    ).structured_content
    assert second is not None
    (row,) = second["results"]["k"]["per_run"]["items"]
    assert set(row) == {"value"} and "rms" in row["value"]


async def test_the_attached_analysis_continues_with_its_own_view(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    """The cursor an attached analysis hands out continues on analyze_results
    with the rows its own page showed, not empty ones."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    block = experiments.AttachedAnalysis.model_validate(
        {
            "recipes": [_RECIPE],
            "all_steps": True,
            "include": {"per_run": {"limit": 1}, "fields": ["rms"]},
        }
    )
    assert block.include is not None
    stored = await capture_attached_analysis(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "label": "rc"}],
                "recipes": [_RECIPE],
                "all_steps": True,
                "include": {"per_run": {"limit": 1}},
            }
        ),
        state_no_sim,
    )
    shown = render_attached_analysis(stored, fields=block.include.fields)
    (first,) = shown["results"]["k"]["per_run"]["items"]
    assert set(first) == {"value"} and "rms" in first["value"]
    continued = (
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate({"continue": shown["next"]}), state_no_sim
        )
    ).structured_content
    assert continued is not None
    (row,) = continued["results"]["k"]["per_run"]["items"]
    assert set(row) == {"value"} and "rms" in row["value"]


async def test_a_cursor_carrying_a_bare_name_reads_it_under_value(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    """A cursor handed out before bare names were resolved still pages rows."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    request = {
        "sources": [{"path": str(raw), "label": "rc"}],
        "recipes": [_RECIPE],
        "all_steps": True,
        "include": {"per_run": {"limit": 1}},
    }
    first = (
        await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state_no_sim)
    ).structured_content
    assert first is not None
    bare = result_store.reencode_cursor(first["next"]["cursor"], view_fields=["rms"])
    continued = (
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {"continue": {"result_set_id": first["result_set_id"], "cursor": bare}}
            ),
            state_no_sim,
        )
    ).structured_content
    assert continued is not None
    (row,) = continued["results"]["k"]["per_run"]["items"]
    assert set(row) == {"value"} and "rms" in row["value"]


# ---------------------------------------------------------------------------
# run_fields: run_experiments and jobs(runs) agree
# ---------------------------------------------------------------------------

#: (case, run_fields, refused, expected row): run rows carry assignments {'R': index}.
_RUN_CASES = [
    ("root", ["case_id"], False, {"case_id": "case-0000"}),
    ("nested", ["assignments.R"], False, {"assignments": {"R": 0}}),
    ("escaped dot", [r"assignments.R\.x"], False, {}),
    ("bare name", ["R"], True, None),
    ("unknown root", ["assignment.R"], True, None),
    ("empty segment", ["assignments."], True, None),
    ("absent leaf", ["assignments.Rx"], False, {}),
    ("repeat", ["case_id", "case_id"], False, {"case_id": "case-0000"}),
]


@pytest.mark.parametrize(
    ("case", "run_fields", "refused", "expected"), _RUN_CASES, ids=[c[0] for c in _RUN_CASES]
)
async def test_run_fields_read_the_same_on_run_experiments_and_jobs_runs(
    state_no_sim: SessionState,
    case: str,
    run_fields: list[str],
    refused: bool,
    expected: dict[str, Any] | None,
) -> None:
    run_request = {"circuits": [{"path": "rc.cir"}], "run_fields": run_fields}
    jobs_request = {"action": "runs", "job_id": "exp_paths_0001", "run_fields": run_fields}
    if refused:
        submitted = _refusal(lambda: experiments.RunExperimentsInput.model_validate(run_request))
        paged = _refusal(lambda: jobs.JobsInput.model_validate(jobs_request))
        assert "run_fields path" in submitted and "run_fields path" in paged
        assert (
            submitted.split("run_fields path", 1)[1].split("[")[0]
            == (paged.split("run_fields path", 1)[1].split("[")[0])
        )
        if case == "unknown root":
            assert "choose one of: " + ", ".join(receipts.RUN_ROW_KEYS) in submitted
        return

    job = make_experiment_job(state_no_sim, job_id="exp_paths_0001", count=1)
    resolved = experiments.RunExperimentsInput.model_validate(run_request).run_fields
    receipt = receipts.render_receipt_snapshot(
        receipts.snapshot_receipt(job, state_no_sim), run_fields=resolved
    )
    page = (
        await jobs.handle_jobs(jobs.JobsInput.model_validate(jobs_request), state_no_sim)
    ).structured_content
    assert page is not None
    assert receipt["runs"]["items"] == page["items"] == [expected]
    assert receipt.get("warnings", []) == page.get("warnings", [])
    if case == "absent leaf":
        (warning,) = page.get("warnings", [])
        assert "run_fields path 'assignments.Rx' is absent" in warning
        assert "keys present at 'assignments': R" in warning
