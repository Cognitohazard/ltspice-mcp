"""Contract tests for the complete Tier-1 Python API door."""

from __future__ import annotations

import copy
import json
import re
import typing
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_args

import pytest
from mcp import types
from pydantic import BaseModel, ValidationError

from ltspice_mcp.api import (
    ApiCallError,
    ApiInternalError,
    ApiInterrupted,
    ApiValidationError,
)
from ltspice_mcp.api import _methods as methods_module
from ltspice_mcp.api._methods import _unwrap
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import recent, services
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments, inspect_tools, schematic_edit, verify
from ltspice_mcp.tools import jobs as jobs_mod
from ltspice_mcp.tools.reference_index import validation_error_detail
from tests.conftest import SyncApi, make_experiment_job, stage_recorded_fixture


def _result(payload: Mapping[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="handler result")],
        structured_content=copy.deepcopy(dict(payload)),
        is_error=is_error,
    )


def test_raw_page_returns_each_handler_payload_verbatim(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = SyncApi(state_no_sim)
    cases = [
        (
            experiments,
            "handle_run_experiments",
            api.run_experiments,
            {"circuits": [{"path": "deck.cir"}], "execution": {"wait_s": 17}},
        ),
        (jobs_mod, "handle_jobs", api.jobs, {"action": "list"}),
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
    api = SyncApi(state_no_sim)
    method = getattr(api, method_name)
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        method(**arguments)


def test_validation_uses_the_server_renderer_and_field_owners(
    state_no_sim: SessionState,
) -> None:
    api = SyncApi(state_no_sim)
    raw = {"action": "status"}
    with pytest.raises(ValidationError) as model_error:
        jobs_mod.JobsInput.model_validate(raw)
    expected = validation_error_detail(
        "jobs",
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


def test_jobs_list_returns_every_circuit_group_the_wire_would_page(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real control plane, with more circuits than one wire page."""
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
    api = SyncApi(state_no_sim)
    expected = []
    for index in range(7):
        circuit = work_dir / f"circuit-{index}.cir"
        circuit.write_text("V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
        recent.touch(circuit)
        expected.append(str(circuit))

    paged = api.jobs(raw_page=True, action="list", limit=2)
    collected = api.jobs(action="list", limit=2)

    # The wire pages at the limit it was given; the Python API renders the
    # same evaluation complete.
    assert paged["returned"] == 2
    assert paged["truncated"] is True
    assert sorted(str(group["path"]) for group in collected["items"]) == sorted(expected)
    assert collected["returned"] == collected["total"] == 7
    assert collected["truncated"] is False
    assert collected["next_cursor"] is None


@pytest.mark.parametrize("action", ["status", "wait", "runs"])
def test_jobs_receipt_is_rendered_from_a_single_read_of_the_job(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    """Both doors render one evaluation, so a receipt describes one read.

    The Python API used to invoke the wire handler and then resolve and
    snapshot the job a second time to render the complete result, keeping
    `timed_out` from the first read and the receipt from the second — two reads
    reported as one answer, with a window in between for the job to move.
    """
    job = make_experiment_job(state_no_sim, job_id="exp-one-read", count=2)
    api = SyncApi(state_no_sim)
    reads: list[str] = []
    resolve = services.resolve_job_async

    async def counting_resolve(job_id: str, state: SessionState):
        reads.append(job_id)
        return await resolve(job_id, state)

    monkeypatch.setattr(services, "resolve_job_async", counting_resolve)
    arguments: dict[str, Any] = {"action": action, "job_id": job.job_id}
    if action == "wait":
        arguments["timeout_s"] = 0
    data = api.jobs(**arguments)

    assert reads == [job.job_id]
    assert data["action"] == action


def test_jobs_error_from_the_in_process_door_carries_the_wire_envelope(
    state_no_sim: SessionState,
) -> None:
    api = SyncApi(state_no_sim)
    with pytest.raises(ApiCallError) as raised:
        api.jobs(action="status", job_id="exp-does-not-exist")
    assert raised.value.code == "job_not_found"
    assert raised.value.payload["action"] == "status"
    assert raised.value.payload["outcome"] == "failed"


def test_run_receipt_assembles_more_than_fifty_runs_with_original_projection(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = SyncApi(state_no_sim)
    job = make_experiment_job(state_no_sim, job_id="exp-many", count=61)
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
    api = SyncApi(state_no_sim)
    job = make_experiment_job(
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
    monkeypatch.setattr(SyncApi, "wait", interrupted_wait)
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
    api = SyncApi(state_no_sim)
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
    api = SyncApi(state_no_sim)
    job = make_experiment_job(
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
    api = SyncApi(state_no_sim)
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
    api = SyncApi(state_no_sim)
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
    api = SyncApi(state_no_sim)
    seen_positions: list[int | None] = []
    seen_requests: list[analyze.AnalyzeResultsInput] = []

    async def evaluate(
        request: analyze.AnalyzeResultsInput,
        _state: SessionState,
        *,
        continuation: analyze.AnalysisContinuationPosition | None = None,
        loaded: object | None = None,
    ):
        del loaded
        seen_positions.append(None if continuation is None else continuation.work_index)
        seen_requests.append(request)
        next_position = len(seen_positions)
        return SimpleNamespace(
            item=None,
            continuation=(
                analyze.AnalysisContinuationPosition("set-1", next_position)
                if next_position < 3
                else None
            ),
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
    api = SyncApi(state_no_sim)
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
    api = SyncApi(state_no_sim)
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
            },
        ],
        all_steps=True,
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
    api = SyncApi(state_no_sim)
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
    )

    async def evaluate_edit(_request: object, _state: SessionState):
        return schematic_edit.EditSchematicEvaluation(
            data={"outcome": "complete", "sha256": "a" * 64},
            text="complete",
            views=views,
        )

    monkeypatch.setattr(verify, "evaluate_verify_circuit", evaluate_verify)
    monkeypatch.setattr(schematic_edit, "evaluate_edit_schematic", evaluate_edit)
    assert api.verify_circuit(path="deck.cir")["findings"] == findings
    # The whole-sheet legend is named explicitly: the default view is now the
    # touched-scope one, which an empty op batch correctly leaves empty. What
    # is under test here is that the API door does not PAGE what it returns.
    edit = api.edit_schematic(target="sheet.asc", ops=[], return_views=["pin_legend"])
    legend = edit["views"]["pin_legend"]
    assert legend["items"] == list(pin_rows)
    assert legend["returned"] == legend["total"] == 121
    assert legend["truncated"] is False
    assert legend["next_cursor"] is None


# ---------------------------------------------------------------------------
# The automatic mode's denylist, pinned fail-closed
# ---------------------------------------------------------------------------

#: A field name that reads like a wire-only paging or budget control. The door
#: classifies by name, so this pattern is what the pin below sweeps for.
_WIRE_ONLY_NAME = re.compile(
    r"^(budget|continue|continuation|cursor|view_cursors|wait_s)$|_cursors?$"
)

#: Wire-only-looking fields the automatic mode deliberately ACCEPTS, each with
#: the reason it is not a paging or budget control. Empty today: every such
#: field on the five input models is rejected. A new one must be added here
#: with its justification, or the interface must reject it — this test fails until
#: one of the two happens, so a paging knob cannot reach the automatic mode by
#: nobody having classified it.
_DOOR_ALLOWLIST: dict[tuple[str, ...], str] = {}

_DOOR_MODELS = (
    experiments.RunExperimentsInput,
    jobs_mod.JobsInput,
    analyze.AnalyzeResultsInput,
    inspect_tools.InspectInput,
    schematic_edit.EditSchematicInput,
    verify.VerifyCircuitInput,
)


def _nested_models(annotation: Any) -> Iterator[type[BaseModel]]:
    """Every model reachable from one field annotation, through unions/lists."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for argument in get_args(annotation):
        yield from _nested_models(argument)


def _field_paths(
    model: type[BaseModel],
    prefix: tuple[str, ...] = (),
    seen: frozenset[type[BaseModel]] = frozenset(),
) -> Iterator[tuple[str, ...]]:
    """Every wire-name path a caller could spell inside this model's arguments."""
    if model in seen:
        return
    for name, field in model.model_fields.items():
        path = (*prefix, field.alias or name)
        yield path
        for nested in _nested_models(field.annotation):
            yield from _field_paths(nested, path, seen | {model})


def _nest(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    for key in reversed(path):
        value = {key: value}
    return value


def _door_rejection(arguments: dict[str, Any]) -> str | None:
    """The door's refusal for these arguments, or None if it lets them through."""
    try:
        methods_module._enforce_auto_door(arguments)
    except ValueError as exc:
        return str(exc)
    return None


def test_every_wire_only_field_is_rejected_or_explicitly_allowlisted() -> None:
    paths = {path for model in _DOOR_MODELS for path in _field_paths(model)}
    candidates = sorted(path for path in paths if _WIRE_ONLY_NAME.search(path[-1]))
    assert candidates, "the sweep found no wire-only fields at all — the walk is broken"

    unclassified: list[str] = []
    unnamed: list[str] = []
    for path in candidates:
        if path in _DOOR_ALLOWLIST:
            continue
        dotted = ".".join(path)
        rejection = _door_rejection(_nest(path, "x"))
        if rejection is None:
            unclassified.append(dotted)
        elif dotted not in rejection:
            unnamed.append(dotted)
    assert not unclassified, (
        "wire-only field(s) reach the automatic mode unclassified: "
        + ", ".join(unclassified)
        + " — reject them in _enforce_auto_door or allowlist them with a reason"
    )
    assert not unnamed, "the interface rejected but did not name: " + ", ".join(unnamed)


class TestAutoDoorRefusalsAreActionable:
    """The refusal a caller who followed the skill actually hits.

    ``budget`` is taught as a first-class knob on four tools and advertised in
    the MCP schema; this door rejects it. That is the intended contract — but the
    refusal has to name the fix for the field it refused, and be catchable by
    the exception the API's own documentation tells callers to catch.
    """

    def test_budget_is_refused_as_a_presentation_cap_not_a_paging_control(self):
        with pytest.raises(ApiValidationError) as caught:
            methods_module._enforce_auto_door({"budget": 4000})
        message = str(caught.value)
        assert "budget" in message
        assert "complete results" in message
        assert "raw_page" not in message, (
            "raw_page returns one handler page instead of the collected result — "
            "a semantic change, and the wrong fix for a presentation cap"
        )

    def test_wait_s_is_refused_by_pointing_at_the_doors_own_dwell(self):
        with pytest.raises(ApiValidationError) as caught:
            methods_module._enforce_auto_door({"execution": {"wait_s": 30}})
        message = str(caught.value)
        assert "execution.wait_s" in message
        assert "api.wait" in message

    def test_a_paging_control_still_gets_the_raw_page_remedy(self):
        with pytest.raises(ApiValidationError) as caught:
            methods_module._enforce_auto_door({"cursor": "o:5"})
        assert "raw_page=True" in str(caught.value)

    def test_the_refusal_is_catchable_as_the_documented_api_error(self):
        """``ApiValidationError`` subclasses ValueError, not the reverse — a bare
        ValueError here slips past every documented ``except ApiValidationError``."""
        with pytest.raises(ApiValidationError):
            methods_module._enforce_auto_door({"budget": 500})


def test_fire_and_forget_receipt_says_the_job_dies_with_this_process(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """close() cancels every job this process owns, so wait=False is not durable.

    The receipt is indistinguishable from a durable submission otherwise, and
    the loss shows up only as a cancellation the caller never requested.
    """
    api = SyncApi(state_no_sim)
    running = make_experiment_job(state_no_sim, job_id="exp-detached", status="running")
    finished = make_experiment_job(state_no_sim, job_id="exp-settled", status="completed")

    async def handler(args: experiments.RunExperimentsInput, _state: SessionState):
        job = running if args.request_id == running.request_id else finished
        return _result(
            {
                "job_id": job.job_id,
                "request_id": job.request_id,
                "control_token": job.control_token,
                "status": job.status,
                "outcome": "in_progress" if job is running else "complete",
            }
        )

    monkeypatch.setattr(experiments, "handle_run_experiments", handler)

    def codes(receipt: Mapping[str, Any]) -> set[str]:
        return {item["code"] for item in receipt["observations"]}

    detached = api.run_experiments(
        wait=False,
        request_id=running.request_id,
        circuits=[{"path": "dut.cir"}],
    )
    settled = api.run_experiments(
        wait=False,
        request_id=finished.request_id,
        circuits=[{"path": "dut.cir"}],
    )
    awaited = api.run_experiments(
        request_id=finished.request_id,
        circuits=[{"path": "dut.cir"}],
    )

    assert "process_owned_job" in codes(detached)
    detail = next(
        item["detail"] for item in detached["observations"] if item["code"] == "process_owned_job"
    )
    assert "long-lived server" in detail
    # A job that is already terminal cannot be lost, and a caller that waited
    # has nothing left to be warned about.
    assert "process_owned_job" not in codes(settled)
    assert "process_owned_job" not in codes(awaited)


def test_dict_shaped_inspect_queries_serialize_without_warning(
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dict query is the documented shape on MCP and the Python API, so the successful
    call must be silent. Dumping the whole request serialized each dict against
    the union member it was declared as and wrote a pydantic serializer warning
    to stderr per query — noise that reads as a malformed call."""
    api = SyncApi(state_no_sim)

    async def handler(args: inspect_tools.InspectInput, _state: SessionState):
        return _result(
            {
                "results": [
                    {"index": index, "ok": True, "kind": "symbol", "data": {}}
                    for index, _query in enumerate(args.queries)
                ]
            }
        )

    monkeypatch.setattr(inspect_tools, "handle_inspect", handler)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        api.inspect(
            queries=[
                {"kind": "symbol", "name": "nmos4"},
                {"kind": "symbol", "name": "pmos4"},
            ]
        )

    assert [str(item.message) for item in caught] == []


# ---------------------------------------------------------------------------
# Relative path arguments are taken from the session's working directory
# ---------------------------------------------------------------------------


@pytest.fixture
def elsewhere(tmp_path_factory, monkeypatch) -> Path:
    """Run with a cwd that is not the session's working directory.

    The contract's documented idiom is ``Api(working_dir=D)`` plus a bare
    filename; while cwd happened to be D the two were indistinguishable.
    """
    away = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(away)
    return away


@pytest.fixture
def state_relative_sandbox(work_dir: Path, elsewhere: Path) -> SessionState:
    """A session configured the way the generated TOML configures one.

    ``[security] allowed_paths = ["."]`` is what the server writes on first
    run, so the sandbox root is itself relative — and left anchored on the
    process cwd it sent every bare filename to the wrong directory.
    """
    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[Path(".")],
        log_level="DEBUG",
    )
    return SessionState.create(config, available={"fake": _FakeSim})


class _FakeSim:
    """Stub simulator class; nothing in these tests reaches an executable."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/sim"]


def _relative_deck(work_dir: Path) -> str:
    (work_dir / "deck.cir").write_text("* d\nV1 in 0 5\nR1 in 0 1k\n.op\n.end\n")
    return "deck.cir"


def test_verify_circuit_takes_a_relative_path_from_the_working_dir(
    state_relative_sandbox: SessionState,
    work_dir: Path,
) -> None:
    api = SyncApi(state_relative_sandbox)
    data = api.verify_circuit(path=_relative_deck(work_dir), checks=["syntax"])
    assert data["path"] == str(work_dir / "deck.cir")


def test_inspect_takes_a_relative_path_from_the_working_dir(
    state_relative_sandbox: SessionState,
    work_dir: Path,
) -> None:
    api = SyncApi(state_relative_sandbox)
    data = api.inspect(queries=[{"kind": "components", "path": _relative_deck(work_dir)}])
    entry = data["results"][0]
    assert entry["ok"] is True, entry
    # Only the working directory holds this deck, so reading it at all is the
    # proof that the relative name was taken from there.
    assert {row["reference"] for row in entry["data"]["components"]} == {"V1", "R1"}


def test_run_experiments_takes_a_relative_circuit_path_from_the_working_dir(
    state_relative_sandbox: SessionState,
    work_dir: Path,
    elsewhere: Path,
) -> None:
    """Pinned on the path the failure names: a deck that is absent under the
    working directory must be reported there, not under the caller's cwd."""
    api = SyncApi(state_relative_sandbox)
    receipt = api.run_experiments(circuits=[{"path": "absent.cir"}], wait=False)
    reported = json.dumps(receipt)
    assert str(work_dir / "absent.cir") in reported
    assert str(elsewhere / "absent.cir") not in reported


def test_the_mcp_door_still_resolves_against_the_process_cwd(
    state_relative_sandbox: SessionState,
    elsewhere: Path,
) -> None:
    """The base is the Python API's opt-in. A server request carries none,
    so the same relative sandbox resolves exactly where it always did."""
    from ltspice_mcp.tools._base import safe_path

    assert safe_path("deck.cir", state_relative_sandbox) == elsewhere / "deck.cir"


def test_inspect_reference_answers_through_the_same_handler(
    state_no_sim: SessionState,
) -> None:
    """The Python API validates ``queries`` with the tool's own model and calls
    the tool's own handler, so a query kind added to the MCP surface has to be
    callable here without the API being taught about it. Pinned because a
    second validation of the kinds — anywhere on the API path — would make the
    two doors disagree about what exists."""
    api = SyncApi(state_no_sim)
    data = api.inspect(queries=[{"kind": "reference", "query": "phase margin"}])
    entry = data["results"][0]
    assert entry["ok"] is True, entry
    top = entry["data"]["matches"][0]
    assert (top["tool"], top["name"]) == ("analyze_results", "stability")
    assert {field["name"] for field in top["fields"]} >= {"key", "signal", "reduce"}
