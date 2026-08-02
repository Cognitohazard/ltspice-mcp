"""Contract tests for the complete Tier-1 Python API door."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Coroutine, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from mcp import types
from pydantic import ValidationError

from ltspice_mcp.api import (
    ApiCallError,
    ApiInternalError,
    ApiInterrupted,
    ApiValidationError,
)
from ltspice_mcp.api import _methods as methods_module
from ltspice_mcp.api._methods import ApiMethodsMixin, _unwrap
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments, inspect_tools, schematic_edit, verify
from tests.conftest import stage_recorded_fixture

_T = TypeVar("_T")


class _SyncApi(ApiMethodsMixin):
    """Small synchronous host that exercises the public mixin over a real state."""

    def __init__(self, state: SessionState) -> None:
        self._state = state

    def _check_process_and_thread(self) -> None:
        return None

    def _call(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        cancelable: bool = False,
        cancel_on_interrupt: bool = False,
        preserve_interrupt: bool = False,
    ) -> _T:
        del cancelable, cancel_on_interrupt, preserve_interrupt
        return asyncio.run(coroutine)


def _result(payload: Mapping[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="handler result")],
        structuredContent=copy.deepcopy(dict(payload)),
        isError=is_error,
    )


def test_raw_page_returns_each_handler_payload_verbatim(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    cases = [
        (
            experiments,
            "handle_run_experiments",
            api.run_experiments,
            {"circuits": [{"path": "deck.cir"}], "execution": {"wait_s": 17}},
        ),
        (experiments, "handle_jobs", api.jobs, {"action": "list"}),
        (
            analyze,
            "handle_analyze_results",
            api.analyze_results,
            {
                "sources": [{"raw_path": "result.raw", "label": "dut"}],
                "recipes": [{"key": "summary", "metric": "summary"}],
                "budget": 500,
            },
        ),
        (
            inspect_tools,
            "handle_inspect",
            api.inspect,
            {"queries": [{"kind": "capabilities"}], "budget": 500},
        ),
        (
            schematic_edit,
            "handle_edit_schematic",
            api.edit_schematic,
            {"target": "sheet.asc", "ops": [], "view_cursors": {}},
        ),
        (verify, "handle_verify_circuit", api.verify_circuit, {"path": "deck.cir"}),
    ]

    for index, (module, handler_name, method, arguments) in enumerate(cases):
        expected = {"operation": handler_name, "page": index}

        async def handler(_args: object, _state: SessionState, data=expected):
            return _result(data)

        monkeypatch.setattr(module, handler_name, handler)
        assert method(raw_page=True, **arguments) == expected


@pytest.mark.parametrize(
    ("method_name", "arguments", "field"),
    [
        ("run_experiments", {"execution": {"wait_s": 1}}, "execution.wait_s"),
        ("run_experiments", {"budget": 500}, "budget"),
        ("jobs", {"action": "list", "cursor": "o:1"}, "cursor"),
        (
            "analyze_results",
            {"continue": {"result_set_id": "set", "cursor": "token"}},
            "continue",
        ),
        (
            "analyze_results",
            {"include": {"per_run": {"cursor": "token"}}},
            "include.per_run.cursor",
        ),
        (
            "inspect",
            {"queries": [{"kind": "symbols", "cursor": "token"}]},
            "queries.cursor",
        ),
        ("edit_schematic", {"view_cursors": {}}, "view_cursors"),
    ],
)
def test_automatic_door_rejects_every_wire_control_before_dispatch(
    state_no_sim: SessionState,
    method_name: str,
    arguments: dict[str, Any],
    field: str,
) -> None:
    api = _SyncApi(state_no_sim)
    method = getattr(api, method_name)
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        method(**arguments)


def test_validation_uses_the_server_renderer_and_field_owners(
    state_no_sim: SessionState,
) -> None:
    api = _SyncApi(state_no_sim)
    raw = {"action": "status"}
    with pytest.raises(ValidationError) as model_error:
        experiments.JobsInput.model_validate(raw)
    expected = compact_validation_error(
        model_error.value,
        field_owners=state_no_sim.field_owners,
    )

    with pytest.raises(ApiValidationError) as api_error:
        api.jobs(action="status")
    assert str(api_error.value) == f"Invalid arguments for jobs: {expected}"


def test_call_error_keeps_payload_handles_and_missing_structured_is_internal() -> None:
    payload = {
        "job_id": "exp-1",
        "control_token": "secret",
        "error": {
            "code": "submission_failed",
            "message": "could not submit",
            "commit_state": "committed",
        },
    }
    with pytest.raises(ApiCallError) as raised:
        _unwrap(_result(payload, is_error=True))
    assert raised.value.payload == payload
    assert raised.value.code == "submission_failed"
    assert raised.value.commit_state == "committed"
    assert raised.value.job_id == "exp-1"
    assert raised.value.control_token == "secret"

    missing = types.CallToolResult(
        content=[types.TextContent(type="text", text="no structured data")]
    )
    with pytest.raises(ApiInternalError, match="structuredContent"):
        _unwrap(missing)


def test_jobs_list_collects_every_flat_page(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    rows = [{"path": f"circuit-{index}.cir"} for index in range(113)]
    calls: list[str | None] = []

    async def handler(args: experiments.JobsInput, _state: SessionState):
        calls.append(args.cursor)
        offset = int(args.cursor.split(":", 1)[1]) if args.cursor else 0
        shown = rows[offset : offset + args.limit]
        next_offset = offset + len(shown)
        payload = {
            "action": "list",
            "outcome": "complete",
            "items": shown,
            "total": len(rows),
            "returned": len(shown),
            "truncated": next_offset < len(rows),
            "next_cursor": f"o:{next_offset}" if next_offset < len(rows) else None,
            "observations": [{"code": "inventory", "detail": "stable"}],
            "warnings": [],
            "failures": [],
            "hint": "page",
        }
        return _result(payload)

    monkeypatch.setattr(experiments, "handle_jobs", handler)
    collected = api.jobs(action="list", limit=17)
    assert collected["items"] == rows
    assert collected["returned"] == collected["total"] == 113
    assert collected["truncated"] is False
    assert collected["next_cursor"] is None
    assert len(calls) == 7
    assert collected["observations"] == [{"code": "inventory", "detail": "stable"}]


def _experiment_job(
    state: SessionState,
    *,
    job_id: str,
    count: int,
    status: str = "completed",
) -> ExperimentJob:
    cases = [
        ExperimentCase(
            case_id=f"case-{index:04d}",
            run_index=index,
            circuit="dut",
            circuit_path=state.working_dir / "dut.cir",
            staged_deck=state.working_dir / "staged.cir",
            deck_sha256="a" * 64,
            assignments={"R": index},
            status="produced" if status == "completed" else "queued",
            raw_file=state.working_dir / f"case-{index}.raw",
            log_file=state.working_dir / f"case-{index}.log",
        )
        for index in range(count)
    ]
    produced = count if status == "completed" else 0
    job = ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="fingerprint",
        canonicalizer_version=1,
        control_token="control-token",
        store_path=state.working_dir / f"{job_id}.json",
        cases=cases,
        sources=[
            SourceRecord(
                circuit="dut",
                path=state.working_dir / "dut.cir",
                sha256="a" * 64,
                staged_deck=state.working_dir / "staged.cir",
                simulator="ltspice",
            )
        ],
        simulator="ltspice",
        completeness=Completeness(
            declared=count,
            expanded=count,
            submitted=produced,
            produced=produced,
        ),
        status=cast(Any, status),
    )
    state.add_experiment_job(job, already_persisted=True)
    return job


def test_run_receipt_assembles_more_than_fifty_runs_with_original_projection(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    job = _experiment_job(state_no_sim, job_id="exp-many", count=61)
    captured: list[experiments.RunExperimentsInput] = []

    async def handler(args: experiments.RunExperimentsInput, _state: SessionState):
        captured.append(args)
        return _result(
            {
                "job_id": job.job_id,
                "request_id": job.request_id,
                "control_token": job.control_token,
                "status": job.status,
                "outcome": "complete",
            }
        )

    monkeypatch.setattr(experiments, "handle_run_experiments", handler)
    common = {
        "request_id": job.request_id,
        "circuits": [{"path": "dut.cir"}],
        "analyze": {
            "recipes": [{"key": "summary", "metric": "summary"}],
            "include": {
                "per_run": {"limit": 7},
                "outliers": True,
                "signals_available": True,
                "fields": ["case_id", "value.mean"],
            },
        },
    }
    lean = api.run_experiments(wait=False, **common)
    projected = api.run_experiments(
        wait=False,
        run_fields=["case_id", "raw"],
        **common,
    )

    assert lean["runs"]["returned"] == lean["runs"]["total"] == 61
    assert lean["runs"]["truncated"] is False
    assert lean["runs"]["next_cursor"] is None
    assert "raw" not in lean["runs"]["items"][0]
    assert set(projected["runs"]["items"][0]) == {"case_id", "raw"}
    direct_runs = api.jobs(action="runs", job_id=job.job_id)
    status = api.jobs(action="status", job_id=job.job_id)
    assert direct_runs["returned"] == direct_runs["total"] == 61
    assert "raw" in direct_runs["items"][0]
    assert status["runs"]["returned"] == status["runs"]["total"] == 61
    assert "raw" not in status["runs"]["items"][0]
    assert captured[0].execution.wait_s == captured[1].execution.wait_s == 0
    assert captured[0].analyze is not None
    assert captured[0].analyze.include is not None
    assert captured[0].analyze.include.per_run is not None
    assert captured[0].analyze.include.per_run.limit == 7
    assert captured[0].analyze.include.outliers is True
    assert captured[0].analyze.include.signals_available is True

    mcp_request = experiments.RunExperimentsInput.model_validate(common)
    assert experiments.canonical_fingerprint(captured[0]) == experiments.canonical_fingerprint(
        mcp_request
    )


def test_interrupt_after_submission_keeps_receipt_and_does_not_cancel_job(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    job = _experiment_job(
        state_no_sim,
        job_id="exp-interrupted-wait",
        count=1,
        status="queued",
    )

    async def handler(_args: experiments.RunExperimentsInput, _state: SessionState):
        return _result(
            {
                "job_id": job.job_id,
                "request_id": job.request_id,
                "control_token": job.control_token,
                "status": "queued",
                "outcome": "in_progress",
            }
        )

    def interrupted_wait(_self: object, job_id: str, timeout: float | None = None):
        del timeout
        raise ApiInterrupted(job_id=job_id)

    monkeypatch.setattr(experiments, "handle_run_experiments", handler)
    monkeypatch.setattr(_SyncApi, "wait", interrupted_wait)
    with pytest.raises(ApiInterrupted) as interrupted:
        api.run_experiments(circuits=[{"path": "dut.cir"}])
    assert interrupted.value.receipt is not None
    assert interrupted.value.receipt["job_id"] == job.job_id
    assert interrupted.value.receipt["control_token"] == job.control_token
    assert state_no_sim.all_jobs[job.job_id].status == "queued"


def test_collector_failure_after_submission_keeps_original_recovery_handles(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    receipt = {
        "job_id": "exp-collector-failed",
        "request_id": "request-collector-failed",
        "control_token": "collector-token",
        "status": "queued",
        "outcome": "in_progress",
    }

    async def handler(_args: experiments.RunExperimentsInput, _state: SessionState):
        return _result(receipt)

    async def fail_collection(*_args: object, **_kwargs: object):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(experiments, "handle_run_experiments", handler)
    monkeypatch.setattr(methods_module, "_complete_run_receipt", fail_collection)
    with pytest.raises(ApiCallError) as raised:
        api.run_experiments(wait=False, circuits=[{"path": "dut.cir"}])
    assert raised.value.payload == receipt
    assert raised.value.job_id == receipt["job_id"]
    assert raised.value.control_token == receipt["control_token"]


def test_wait_timeout_returns_snapshot_and_leaves_job_running(
    state_no_sim: SessionState,
) -> None:
    api = _SyncApi(state_no_sim)
    job = _experiment_job(
        state_no_sim,
        job_id="exp-running",
        count=2,
        status="queued",
    )
    waited = api.wait(job.job_id, timeout=0)
    assert waited["timed_out"] is True
    assert waited["status"] == "queued"
    assert state_no_sim.all_jobs[job.job_id].status == "queued"


def _inspect_page_item(
    index: int,
    kind: str,
    data: dict[str, Any],
    collections: dict[str, tuple[int, int]],
    cursor: str | None,
) -> dict[str, Any]:
    page_collections = {
        name: {"total": total, "returned": returned, "truncated": returned < total}
        for name, (total, returned) in collections.items()
    }
    return {
        "index": index,
        "kind": kind,
        "ok": True,
        "data": data,
        "next_cursor": cursor,
        "page": {
            "total": sum(value[0] for value in collections.values()),
            "returned": sum(value[1] for value in collections.values()),
            "truncated": cursor is not None,
            "collections": page_collections,
        },
    }


def test_inspect_batches_live_cursors_and_merges_paired_net_collections(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    batch_sizes: list[int] = []

    async def handler(args: inspect_tools.InspectInput, _state: SessionState):
        batch_sizes.append(len(args.queries))
        results: list[dict[str, Any]] = []
        for local_index, query in enumerate(args.queries):
            raw = cast(dict[str, Any], query)
            if raw["kind"] == "components":
                second = raw.get("cursor") == "components-1"
                results.append(
                    _inspect_page_item(
                        local_index,
                        "components",
                        {
                            "components": [{"reference": "R2" if second else "R1"}],
                            "detail": "list",
                            "prefix": None,
                            "total": 2,
                            "returned": 1,
                        },
                        {"components": (2, 1)},
                        None if second else "components-1",
                    )
                )
                continue
            cursor = raw.get("cursor")
            page_number = {None: 0, "net-1": 1, "net-2": 2}[cursor]
            pins = [{"pin": page_number}] if page_number < 2 else []
            coordinates = [{"x": page_number, "y": page_number}]
            results.append(
                _inspect_page_item(
                    local_index,
                    "net",
                    {
                        "source": "schematic",
                        "labels": ["vout"],
                        "pins": pins,
                        "total_pins": 2,
                        "returned": len(pins),
                        "coordinates": coordinates,
                        "total_coordinates": 3,
                        "returned_coordinates": 1,
                        "coordinates_truncated": page_number < 2,
                    },
                    {"pins": (2, len(pins)), "coordinates": (3, 1)},
                    ["net-1", "net-2", None][page_number],
                )
            )
        return _result(inspect_tools.inspect_envelope(results))

    monkeypatch.setattr(inspect_tools, "handle_inspect", handler)
    collected = api.inspect(
        queries=[
            {"kind": "components", "path": "sheet.asc"},
            {"kind": "net", "path": "sheet.asc", "at": "net:vout"},
        ]
    )
    assert batch_sizes == [2, 2, 1]
    components, net = collected["results"]
    assert [row["reference"] for row in components["data"]["components"]] == ["R1", "R2"]
    assert len(net["data"]["pins"]) == 2
    assert len(net["data"]["coordinates"]) == 3
    assert net["data"]["labels"] == ["vout"]
    assert net["page"] == {
        "total": 5,
        "returned": 5,
        "truncated": False,
        "collections": {
            "pins": {"total": 2, "returned": 2, "truncated": False},
            "coordinates": {"total": 3, "returned": 3, "truncated": False},
        },
    }


def test_inspect_stale_restart_discards_the_first_revision(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    calls = 0

    async def handler(args: inspect_tools.InspectInput, _state: SessionState):
        nonlocal calls
        calls += 1
        raw = cast(dict[str, Any], args.queries[0])
        cursor = raw.get("cursor")
        if cursor == "old-next":
            return _result(
                inspect_tools.inspect_envelope(
                    [
                        {
                            "index": 0,
                            "kind": "components",
                            "ok": False,
                            "error": {"code": "invalid_cursor", "message": "stale"},
                        }
                    ]
                )
            )
        restarted = calls >= 3
        rows = (
            [{"reference": "NEW2"}]
            if cursor == "new-next"
            else [{"reference": "NEW1" if restarted else "OLD1"}]
        )
        next_cursor = None if cursor == "new-next" else ("new-next" if restarted else "old-next")
        return _result(
            inspect_tools.inspect_envelope(
                [
                    _inspect_page_item(
                        0,
                        "components",
                        {
                            "components": rows,
                            "detail": "list",
                            "prefix": None,
                            "total": 2,
                            "returned": 1,
                        },
                        {"components": (2, 1)},
                        next_cursor,
                    )
                ]
            )
        )

    monkeypatch.setattr(inspect_tools, "handle_inspect", handler)
    collected = api.inspect(queries=[{"kind": "components", "path": "sheet.asc"}])
    assert [row["reference"] for row in collected["results"][0]["data"]["components"]] == [
        "NEW1",
        "NEW2",
    ]
    assert calls == 4


def test_analyze_drives_neutral_continuations_without_flipping_request_fields(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    seen_positions: list[int | None] = []
    seen_requests: list[analyze.AnalyzeResultsInput] = []

    async def evaluate(
        request: analyze.AnalyzeResultsInput,
        _state: SessionState,
        *,
        continuation: analyze.AnalysisContinuationPosition | None = None,
    ):
        seen_positions.append(None if continuation is None else continuation.work_index)
        seen_requests.append(request)
        next_position = len(seen_positions)
        return SimpleNamespace(
            continuation=(
                analyze.AnalysisContinuationPosition("set-1", next_position)
                if next_position < 3
                else None
            )
        )

    def complete(drives: list[object]) -> dict[str, Any]:
        return {"outcome": "complete", "drives": len(drives)}

    monkeypatch.setattr(analyze, "evaluate_analysis_results", evaluate)
    monkeypatch.setattr(analyze, "complete_analysis_evaluations", complete)
    result = api.analyze_results(
        sources=[{"raw_path": "result.raw", "label": "dut"}],
        recipes=[
            {"key": "summary", "metric": "summary"},
            {"key": "measurements", "metric": "measurements"},
        ],
        include={
            "per_run": {"limit": 7},
            "outliers": True,
            "signals_available": True,
            "fields": ["case_id", "value.mean"],
        },
    )
    assert result == {"outcome": "complete", "drives": 3}
    assert seen_positions == [None, 1, 2]
    assert all(
        request.include.per_run is not None and request.include.per_run.limit == 7
        for request in seen_requests
    )
    assert all(request.include.outliers is True for request in seen_requests)
    assert all(request.include.signals_available is True for request in seen_requests)
    assert all(request.include.fields == ["case_id", "value.mean"] for request in seen_requests)
    assert all(len(request.recipes or []) == 2 for request in seen_requests)


def test_analyze_complete_failure_inventory_reconciles_the_wire_cap(
    state_no_sim: SessionState,
    work_dir: Path,
) -> None:
    api = _SyncApi(state_no_sim)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    recipes = [{"key": f"invalid-{index}", "metric": "not_a_recipe"} for index in range(107)]
    sources = [{"raw_path": str(raw), "label": "dut"}]
    complete = api.analyze_results(sources=sources, recipes=recipes)
    wire = api.analyze_results(raw_page=True, sources=sources, recipes=recipes)

    assert len(complete["failures"]) == 107
    assert complete["next"] is None
    assert not any(item.get("code") == "failures_truncated" for item in complete["observations"])
    assert len(wire["failures"]) == 100
    assert any(
        item.get("code") == "failures_truncated" and "107" in item.get("detail", "")
        for item in wire["observations"]
    )


def test_analyze_collects_projected_per_run_rows_and_missing_cases_together(
    state_no_sim: SessionState,
    work_dir: Path,
) -> None:
    api = _SyncApi(state_no_sim)
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    data = api.analyze_results(
        sources=[{"raw_path": str(raw), "label": "dut", "runs": [0, 3]}],
        recipes=[
            {"key": "summary", "metric": "summary"},
            {
                "key": "values",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "all_steps": True,
            },
        ],
        include={
            "per_run": {"limit": 1},
            "fields": ["case_id", "run_index", "step_index"],
        },
    )
    assert set(data["results"]) == {"summary", "values"}
    values = data["results"]["values"]["per_run"]
    assert values["returned"] == values["total"] == 3
    assert values["truncated"] is False
    assert values["next_cursor"] is None
    assert all(set(row) <= {"case_id", "run_index", "step_index"} for row in values["items"])
    missing = data["coverage"]["missing_cases"]
    assert missing["returned"] == missing["total"] == 1
    assert missing["items"][0]["run_index"] == 3
    assert missing["truncated"] is False
    assert missing["next_cursor"] is None
    assert data["next"] is None


def test_verify_and_edit_return_uncapped_neutral_data(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _SyncApi(state_no_sim)
    findings = [{"rule_id": "spacing", "subject": str(index)} for index in range(31)]

    async def evaluate_verify(_request: object, _state: SessionState):
        return verify.VerifyCircuitEvaluation(
            data={"outcome": "partial", "findings": findings, "failures": []}
        )

    pin_rows = tuple({"reference": f"R{index}"} for index in range(121))
    views = schematic_edit.EditSchematicViews(
        sha256="a" * 64,
        wiring_profile={
            "wire_segments": 0,
            "pins_total": 121,
            "pins_wired": 0,
            "pins_label_only": 0,
        },
        pin_legend=pin_rows,
        label_only_pins=(),
        render=None,
        failures=(),
    )

    async def evaluate_edit(_request: object, _state: SessionState):
        return schematic_edit.EditSchematicEvaluation(
            data={"outcome": "complete", "sha256": "a" * 64},
            text="complete",
            format=None,
            views=views,
        )

    monkeypatch.setattr(verify, "evaluate_verify_circuit", evaluate_verify)
    monkeypatch.setattr(schematic_edit, "evaluate_edit_schematic", evaluate_edit)
    assert api.verify_circuit(path="deck.cir")["findings"] == findings
    edit = api.edit_schematic(target="sheet.asc", ops=[])
    legend = edit["views"]["pin_legend"]
    assert legend["items"] == list(pin_rows)
    assert legend["returned"] == legend["total"] == 121
    assert legend["truncated"] is False
    assert legend["next_cursor"] is None
