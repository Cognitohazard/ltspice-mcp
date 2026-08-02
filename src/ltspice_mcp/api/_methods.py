"""Tier-1 synchronous methods over the consolidated engine operations."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Iterator, Mapping, Sequence
from typing import Any, TypeVar

from mcp import types
from pydantic import BaseModel, ValidationError

from ltspice_mcp.api._exceptions import (
    ApiCallError,
    ApiInternalError,
    ApiInterrupted,
    ApiValidationError,
)
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments, inspect_tools, schematic_edit, verify

_T = TypeVar("_T")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _validation_error(
    operation: str,
    exc: ValidationError,
    state: SessionState,
) -> ApiValidationError:
    detail = compact_validation_error(exc, field_owners=state.field_owners)
    return ApiValidationError(f"Invalid arguments for {operation}: {detail}")


def _validate(
    operation: str,
    model: type[_ModelT],
    arguments: Mapping[str, Any],
    state: SessionState,
) -> _ModelT:
    try:
        return model.model_validate(dict(arguments))
    except ValidationError as exc:
        raise _validation_error(operation, exc, state) from None


def _walk_fields(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            yield child_path
            yield from _walk_fields(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_fields(child, path)


def _enforce_auto_door(arguments: Mapping[str, Any]) -> None:
    rejected: list[str] = []
    for path in _walk_fields(arguments):
        key = path[-1]
        dotted = ".".join(path)
        if (
            key == "budget"
            or key in {"cursor", "continuation", "continue", "view_cursors"}
            or key.endswith("_cursor")
            or key.endswith("_cursors")
            or path == ("execution", "wait_s")
        ):
            rejected.append(dotted)
    if rejected:
        fields = ", ".join(dict.fromkeys(rejected))
        raise ValueError(
            f"Wire-only control(s) are not accepted in automatic mode: {fields}; "
            "pass raw_page=True to request exactly one handler page"
        )


def _message_for_error(payload: Mapping[str, Any], result: types.CallToolResult) -> str:
    error = payload.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message
    for content in result.content:
        if isinstance(content, types.TextContent):
            return content.text
    return "The engine returned a call-level error"


def _unwrap(result: types.CallToolResult) -> dict[str, Any]:
    structured = result.structuredContent
    if not isinstance(structured, Mapping):
        raise ApiInternalError("The engine response did not contain structuredContent")
    payload = copy.deepcopy(dict(structured))
    if result.isError:
        raise ApiCallError(_message_for_error(payload, result), payload=payload)
    return payload


async def _handler_page(
    handler: Any,
    model: BaseModel,
    state: SessionState,
) -> dict[str, Any]:
    return _unwrap(await handler(model, state))


def _collector_error(receipt: Mapping[str, Any], exc: BaseException) -> ApiCallError:
    return ApiCallError(
        f"Result collection failed after durable submission: {exc}",
        payload=receipt,
    )


async def _resolve_snapshot_job(
    job_id: str,
    state: SessionState,
) -> experiments.Job:
    job = state.all_jobs.get(job_id)
    if job is not None:
        return job
    from ltspice_mcp.lib import services

    return await services.resolve_job_async(job_id, state)


async def _complete_run_receipt(
    receipt: Mapping[str, Any],
    request: experiments.RunExperimentsInput,
    state: SessionState,
) -> dict[str, Any]:
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str):
        return copy.deepcopy(dict(receipt))
    job = await _resolve_snapshot_job(job_id, state)
    control_token = receipt.get("control_token")
    snapshot = experiments.snapshot_receipt(
        job,
        state,
        control_token=control_token if isinstance(control_token, str) else None,
    )
    analysis_fields = (
        request.analyze.include.fields
        if request.analyze is not None and request.analyze.include is not None
        else None
    )
    data = experiments.render_receipt_snapshot(
        snapshot,
        control_token=control_token if isinstance(control_token, str) else None,
        provenance=request.provenance,
        run_fields=request.run_fields,
        runs_cap=max(1, len(snapshot.runs_by_key)),
        analysis_fields=analysis_fields,
    )
    return experiments.finalize_receipt(data)


async def _collect_analysis(
    request: analyze.AnalyzeResultsInput,
    state: SessionState,
) -> dict[str, Any]:
    drives: list[analyze.AnalysisEvaluation] = []
    position: analyze.AnalysisContinuationPosition | None = None
    seen: set[tuple[str, int, int, int]] = set()
    while True:
        drive = await analyze.evaluate_analysis_results(
            request,
            state,
            continuation=position,
        )
        drives.append(drive)
        position = drive.continuation
        if position is None:
            return analyze.complete_analysis_evaluations(drives)
        marker = (
            position.result_set_id,
            position.work_index,
            position.row_offset,
            position.missing_offset,
        )
        if marker in seen:
            raise ApiInternalError("The analysis evaluator continuation made no progress")
        seen.add(marker)


def _dedupe(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(copy.deepcopy(value))
    return result


_COLLECTION_COUNTERS: dict[str, tuple[str, str, str | None]] = {
    "symbols": ("total", "returned", None),
    "members": ("total_members", "returned", None),
    "pins": ("total_pins", "returned", None),
    "coordinates": ("total_coordinates", "returned_coordinates", "coordinates_truncated"),
    "components": ("total", "returned", None),
    "results": ("total", "returned", None),
}


def _merge_inspect_item(
    accumulated: dict[str, Any] | None,
    page_item: Mapping[str, Any],
    original_index: int,
) -> dict[str, Any]:
    current = copy.deepcopy(dict(page_item))
    current["index"] = original_index
    if accumulated is None or not current.get("ok"):
        return current

    old_data = accumulated.get("data")
    new_data = current.get("data")
    page = current.get("page")
    if not isinstance(old_data, Mapping) or not isinstance(new_data, dict):
        return current
    if not isinstance(page, dict) or not isinstance(page.get("collections"), dict):
        return current

    collection_names = set(page["collections"])
    for key, old_value in old_data.items():
        if key in collection_names:
            old_rows = old_value if isinstance(old_value, list) else []
            new_rows = new_data.get(key)
            new_data[key] = [
                *copy.deepcopy(old_rows),
                *(copy.deepcopy(new_rows) if isinstance(new_rows, list) else []),
            ]
        elif isinstance(old_value, list) and isinstance(new_data.get(key), list):
            new_data[key] = _dedupe([*old_value, *new_data[key]])

    for collection, metadata in page["collections"].items():
        if not isinstance(metadata, dict):
            continue
        rows = new_data.get(collection)
        returned = len(rows) if isinstance(rows, list) else int(metadata.get("returned", 0))
        metadata["returned"] = returned
        metadata["truncated"] = returned < int(metadata.get("total", returned))
        counters = _COLLECTION_COUNTERS.get(collection)
        if counters is not None:
            total_key, returned_key, truncated_key = counters
            new_data[total_key] = metadata.get("total", returned)
            new_data[returned_key] = returned
            if truncated_key is not None:
                new_data[truncated_key] = metadata["truncated"]

    collections = [value for value in page["collections"].values() if isinstance(value, dict)]
    page["total"] = sum(int(value.get("total", 0)) for value in collections)
    page["returned"] = sum(int(value.get("returned", 0)) for value in collections)
    page["truncated"] = current.get("next_cursor") is not None
    return current


async def _collect_inspect(
    request: inspect_tools.InspectInput,
    state: SessionState,
) -> dict[str, Any]:
    initial = await _handler_page(inspect_tools.handle_inspect, request, state)
    raw_queries = request.model_dump(mode="json")["queries"]
    accumulated: list[dict[str, Any] | None] = [None] * len(raw_queries)
    cursors: dict[int, str | None] = {}
    restarted: set[int] = set()

    for index, item in enumerate(initial["results"]):
        accumulated[index] = _merge_inspect_item(None, item, index)
        cursor = item.get("next_cursor") if item.get("ok") else None
        if isinstance(cursor, str):
            cursors[index] = cursor

    while cursors:
        active = list(cursors)
        query_page: list[dict[str, Any]] = []
        for index in active:
            query = copy.deepcopy(raw_queries[index])
            cursor = cursors[index]
            if cursor is not None:
                query["cursor"] = cursor
            else:
                query.pop("cursor", None)
            query_page.append(query)
        continued = inspect_tools.InspectInput.model_validate({"queries": query_page})
        page = await _handler_page(inspect_tools.handle_inspect, continued, state)
        cursors = {}

        for local_index, page_item in enumerate(page["results"]):
            original_index = active[local_index]
            error = page_item.get("error")
            invalid_cursor = (
                not page_item.get("ok")
                and isinstance(error, Mapping)
                and error.get("code") == "invalid_cursor"
            )
            if invalid_cursor:
                if original_index in restarted:
                    page_payload = copy.deepcopy(page)
                    page_payload["results"][local_index]["index"] = original_index
                    raise ApiCallError(
                        "An inspect query changed again while its complete result was collected",
                        payload=page_payload,
                    )
                restarted.add(original_index)
                accumulated[original_index] = None
                cursors[original_index] = None
                continue

            accumulated[original_index] = _merge_inspect_item(
                accumulated[original_index],
                page_item,
                original_index,
            )
            cursor = page_item.get("next_cursor") if page_item.get("ok") else None
            if isinstance(cursor, str):
                cursors[original_index] = cursor

    results = [
        item
        if item is not None
        else {
            "index": index,
            "kind": None,
            "ok": False,
            "error": {"code": "internal_error", "message": "No inspect result was collected"},
        }
        for index, item in enumerate(accumulated)
    ]
    return inspect_tools.inspect_envelope(results)


async def _collect_jobs_list(
    request: experiments.JobsInput,
    state: SessionState,
) -> dict[str, Any]:
    page = await _handler_page(experiments.handle_jobs, request, state)
    rows = list(page.get("items", []))
    observations = list(page.get("observations", []))
    warnings = list(page.get("warnings", []))
    failures = list(page.get("failures", []))
    cursor = page.get("next_cursor")
    final = page
    while isinstance(cursor, str):
        arguments: dict[str, Any] = {
            "action": "list",
            "limit": request.limit,
            "cursor": cursor,
        }
        if request.circuit is not None:
            arguments["circuit"] = request.circuit
        continued = experiments.JobsInput.model_validate(arguments)
        final = await _handler_page(experiments.handle_jobs, continued, state)
        rows.extend(final.get("items", []))
        observations = _dedupe([*observations, *final.get("observations", [])])
        warnings = _dedupe([*warnings, *final.get("warnings", [])])
        failures = _dedupe([*failures, *final.get("failures", [])])
        cursor = final.get("next_cursor")
    data = copy.deepcopy(final)
    data.update(experiments.unpaged_jobs_items(rows))
    data["observations"] = observations
    data["warnings"] = warnings
    data["failures"] = failures
    return data


async def _collect_jobs_snapshot(
    request: experiments.JobsInput,
    state: SessionState,
) -> dict[str, Any]:
    invoked = await _handler_page(experiments.handle_jobs, request, state)
    if request.action == "cancel":
        return invoked
    if request.action == "list":
        raise ApiInternalError("jobs(list) was routed to the receipt collector")

    job_id = invoked.get("job_id")
    if not isinstance(job_id, str):
        raise ApiInternalError("A successful jobs response did not identify its job")
    job = await _resolve_snapshot_job(job_id, state)
    snapshot = experiments.snapshot_receipt(job, state)
    if request.action == "runs":
        page = experiments.project_receipt_runs(
            snapshot,
            None,
            lean_default=False,
            limit=None,
        )
        return {
            "action": "runs",
            "outcome": snapshot.outcome,
            "job_id": snapshot.job_id,
            "request_id": snapshot.request_id,
            "status": snapshot.status,
            "dialect": snapshot.dialect,
            **page,
            "observations": [],
            "warnings": [],
            "failures": [],
            "hint": f"Returned all recorded runs for job {snapshot.job_id}.",
        }

    timed_out = invoked.get("timed_out") if request.action == "wait" else None
    return experiments.render_jobs_receipt_snapshot(
        request.action,
        snapshot,
        timed_out=timed_out if isinstance(timed_out, bool) else None,
        runs_cap=max(1, len(snapshot.runs_by_key)),
    )


class ApiMethodsMixin(ABC):
    """Public consolidated methods mixed into :class:`ltspice_mcp.api.Api`."""

    _state: SessionState

    @abstractmethod
    def _check_process_and_thread(self) -> None:
        """Reject calls from an inherited process or the private loop thread."""

    @abstractmethod
    def _call(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        cancelable: bool = False,
        cancel_on_interrupt: bool = False,
        preserve_interrupt: bool = False,
    ) -> _T:
        """Marshal one coroutine onto the private loop."""

    def run_experiments(
        self,
        *,
        wait: bool = True,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Submit an experiment and optionally wait for its complete receipt."""
        self._check_process_and_thread()
        if not isinstance(wait, bool):
            raise TypeError("wait must be a bool")
        if raw_page:
            request = _validate(
                "run_experiments", experiments.RunExperimentsInput, arguments, self._state
            )
            return self._call(
                _handler_page(experiments.handle_run_experiments, request, self._state)
            )

        _enforce_auto_door(arguments)
        submitted_arguments = copy.deepcopy(arguments)
        execution = submitted_arguments.get("execution")
        if execution is None:
            submitted_arguments["execution"] = {"wait_s": 0}
        elif isinstance(execution, Mapping):
            submitted_arguments["execution"] = {**execution, "wait_s": 0}
        request = _validate(
            "run_experiments",
            experiments.RunExperimentsInput,
            submitted_arguments,
            self._state,
        )
        receipt = self._call(
            _handler_page(experiments.handle_run_experiments, request, self._state),
            preserve_interrupt=True,
        )
        try:
            complete = self._call(_complete_run_receipt(receipt, request, self._state))
        except KeyboardInterrupt as exc:
            raise ApiInterrupted(receipt=receipt) from exc
        except Exception as exc:
            raise _collector_error(receipt, exc) from exc

        job_id = receipt.get("job_id")
        if not wait or not isinstance(job_id, str) or complete.get("outcome") != "in_progress":
            return complete
        try:
            self.wait(job_id)
            return self._call(_complete_run_receipt(receipt, request, self._state))
        except ApiInterrupted as exc:
            raise ApiInterrupted(receipt=receipt, job_id=job_id) from exc
        except KeyboardInterrupt as exc:
            raise ApiInterrupted(receipt=receipt, job_id=job_id) from exc
        except Exception as exc:
            raise _collector_error(receipt, exc) from exc

    def jobs(self, *, raw_page: bool = False, **arguments: Any) -> dict[str, Any]:
        """Control jobs, collecting list and receipt pages in automatic mode."""
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate("jobs", experiments.JobsInput, arguments, self._state)
        if raw_page:
            return self._call(
                _handler_page(experiments.handle_jobs, request, self._state),
                cancelable=request.action != "cancel",
                cancel_on_interrupt=request.action == "wait",
            )
        if request.action == "list":
            return self._call(
                _collect_jobs_list(request, self._state),
                cancelable=True,
                cancel_on_interrupt=True,
            )
        return self._call(
            _collect_jobs_snapshot(request, self._state),
            cancelable=request.action != "cancel",
            cancel_on_interrupt=request.action == "wait",
        )

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Wait for a job, returning a complete snapshot on terminality or timeout."""
        self._check_process_and_thread()
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
                raise ValueError("timeout must be a non-negative number or None")
            if timeout < 0:
                raise ValueError("timeout must be a non-negative number or None")

        started = time.monotonic()
        while True:
            remaining = (
                None
                if timeout is None
                else max(0.0, float(timeout) - (time.monotonic() - started))
            )
            dwell = (
                experiments.JOBS_WAIT_CAP_S
                if remaining is None
                else min(experiments.JOBS_WAIT_CAP_S, remaining)
            )
            try:
                result = self.jobs(
                    action="wait",
                    job_id=job_id,
                    timeout_s=dwell,
                )
            except ApiInterrupted:
                raise
            except KeyboardInterrupt as exc:
                raise ApiInterrupted(job_id=job_id) from exc
            if not result.get("timed_out"):
                return result
            if timeout is not None and time.monotonic() - started >= float(timeout):
                result["timed_out"] = True
                return result

    def analyze_results(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Analyze results, driving bounded neutral evaluation to completion."""
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate("analyze_results", analyze.AnalyzeResultsInput, arguments, self._state)
        coroutine = (
            _handler_page(analyze.handle_analyze_results, request, self._state)
            if raw_page
            else _collect_analysis(request, self._state)
        )
        return self._call(
            coroutine,
            cancelable=True,
            cancel_on_interrupt=True,
        )

    def inspect(self, *, raw_page: bool = False, **arguments: Any) -> dict[str, Any]:
        """Run batched read queries, collecting every per-query cursor."""
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate("inspect", inspect_tools.InspectInput, arguments, self._state)
        coroutine = (
            _handler_page(inspect_tools.handle_inspect, request, self._state)
            if raw_page
            else _collect_inspect(request, self._state)
        )
        return self._call(coroutine, cancelable=True, cancel_on_interrupt=True)

    def edit_schematic(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Apply one schematic transaction and return its complete in-memory views."""
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate(
            "edit_schematic", schematic_edit.EditSchematicInput, arguments, self._state
        )
        if raw_page:
            return self._call(
                _handler_page(schematic_edit.handle_edit_schematic, request, self._state)
            )

        async def evaluate() -> dict[str, Any]:
            result = await schematic_edit.evaluate_edit_schematic(request, self._state)
            return schematic_edit.complete_edit_schematic_data(result, request)

        return self._call(evaluate())

    def verify_circuit(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Verify a circuit, returning every finding from the neutral evaluator."""
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate("verify_circuit", verify.VerifyCircuitInput, arguments, self._state)
        if raw_page:
            return self._call(_handler_page(verify.handle_verify_circuit, request, self._state))

        async def evaluate() -> dict[str, Any]:
            result = await verify.evaluate_verify_circuit(request, self._state)
            payload = copy.deepcopy(result.data)
            if result.is_error:
                rendered = verify.render_verify_circuit(result)
                raise ApiCallError(_message_for_error(payload, rendered), payload=payload)
            return payload

        return self._call(evaluate(), cancelable=True, cancel_on_interrupt=True)
