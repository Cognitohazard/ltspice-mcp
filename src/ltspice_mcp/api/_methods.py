"""Tier-1 synchronous methods over the consolidated engine operations."""

from __future__ import annotations

import copy
import os
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ltspice_mcp.api import _detach, _reference
from ltspice_mcp.api._exceptions import (
    ApiCallError,
    ApiInternalError,
    ApiInterrupted,
    ApiValidationError,
)
from ltspice_mcp.api._primitives import RawResult, load_measurement_results, load_raw_result
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib import services
from ltspice_mcp.lib.pathutil import relative_paths_from
from ltspice_mcp.state import SessionState

if TYPE_CHECKING:
    from mcp import types

    from ltspice_mcp.tools import (
        analyze,
        experiments,
        inspect_tools,
        jobs,
        schematic_edit,
        verify,
    )
else:
    # Deferred imports (PEP 562-style proxies): the tool modules pull the MCP
    # SDK and the analysis chain — over a second of import an Api() that only
    # reads raws never needs. Each proxy resolves its module on first
    # attribute access; the TYPE_CHECKING branch above keeps pyright's view
    # identical to eager imports. Pinned by the cold-subprocess test in
    # tests/test_api_reference.py.
    class _DeferredModule:
        def __init__(self, dotted: str) -> None:
            self._dotted = dotted

        def __getattr__(self, name: str) -> Any:
            # Resolved on EVERY access, never cached: a cached attribute
            # would pin the value seen first and silently bypass a later
            # monkeypatch on the real module. After the first import this
            # is a sys.modules dict hit plus a getattr.
            module = import_module(self._dotted)
            ensure_method_docs()
            return getattr(module, name)

    analyze = _DeferredModule("ltspice_mcp.tools.analyze")
    experiments = _DeferredModule("ltspice_mcp.tools.experiments")
    inspect_tools = _DeferredModule("ltspice_mcp.tools.inspect_tools")
    jobs = _DeferredModule("ltspice_mcp.tools.jobs")
    schematic_edit = _DeferredModule("ltspice_mcp.tools.schematic_edit")
    verify = _DeferredModule("ltspice_mcp.tools.verify")

_T = TypeVar("_T")
_ModelT = TypeVar("_ModelT", bound=BaseModel)

#: One operation's automatic mode: the complete result, assembled from as many
#: handler pages as it takes.
_Collector = Callable[[Any, SessionState], Coroutine[Any, Any, dict[str, Any]]]

#: Whether a call may be cancelled, either fixed or decided from its request.
_CancelPolicy = bool | Callable[[Any], bool]


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


#: Why each class of wire-only control is refused here, and what to do instead.
#: One remedy per class rather than one for all three: ``raw_page=True`` is the
#: right answer for a pagination control and a semantic change for the other
#: two, and a refusal that hands back the wrong fix costs a retry that ends
#: somewhere worse than where it started.
_DOOR_REMEDIES: dict[str, str] = {
    "budget": (
        "budget is an MCP presentation cap; the Python API returns complete "
        "results — remove the field"
    ),
    "dwell": (
        "execution.wait_s is MCP's response dwell; the Python API already "
        "blocks — use wait=False for a fire-and-forget receipt, or api.wait(job_id)"
    ),
    "paging": (
        "pagination controls belong to a single handler page; the Python API collects "
        "every page — remove them, or pass raw_page=True to drive paging yourself"
    ),
}


def _door_class(path: tuple[str, ...]) -> str | None:
    """Which refusal class this argument path falls in, or None if it is fine."""
    key = path[-1]
    if key == "budget":
        return "budget"
    if path == ("execution", "wait_s"):
        return "dwell"
    if (
        key in {"cursor", "continuation", "continue", "view_cursors"}
        or key.endswith("_cursor")
        or key.endswith("_cursors")
    ):
        return "paging"
    return None


def _enforce_auto_door(arguments: Mapping[str, Any]) -> None:
    rejected: dict[str, list[str]] = {}
    for path in _walk_fields(arguments):
        kind = _door_class(path)
        if kind is not None:
            rejected.setdefault(kind, []).append(".".join(path))
    if not rejected:
        return
    parts = [
        f"{', '.join(dict.fromkeys(fields))}: {_DOOR_REMEDIES[kind]}"
        for kind, fields in rejected.items()
    ]
    raise ApiValidationError(
        "Wire-only control(s) are not accepted in automatic mode. " + "; ".join(parts)
    )


async def _anchored_on(base: Path, coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run one call with its relative path arguments taken from ``base``.

    Applied inside the coroutine for the same reason the automatic-mode flag is:
    the path chokepoint reads the base from the engine loop's task context, and
    a set on the calling thread would never reach it.
    """
    with relative_paths_from(base):
        return await coroutine


async def _through_auto_door(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run one automatic-mode coroutine with the interface marked for the handlers.

    Marked inside the coroutine, not around the ``_call`` that marshals it: the
    handlers read the flag from the engine loop's task context, and a set on the
    calling thread would never reach it.
    """
    from ltspice_mcp.tools._base import automatic_door  # deferred with the tool modules

    with automatic_door():
        return await coroutine


def _payload_message(payload: Mapping[str, Any]) -> str | None:
    """The message an error envelope carries, if it carries one."""
    error = payload.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return None


def _message_for_error(payload: Mapping[str, Any], result: types.CallToolResult) -> str:
    carried = _payload_message(payload)
    if carried is not None:
        return carried
    from mcp import types as mcp_types  # already loaded: a result exists to unwrap

    for content in result.content:
        if isinstance(content, mcp_types.TextContent):
            return content.text
    return "The engine returned a call-level error"


def _unwrap(result: types.CallToolResult) -> dict[str, Any]:
    structured = result.structured_content
    if not isinstance(structured, Mapping):
        raise ApiInternalError("The engine response did not contain structuredContent")
    payload = copy.deepcopy(dict(structured))
    if result.is_error:
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


async def _complete_run_receipt(
    receipt: Mapping[str, Any],
    request: experiments.RunExperimentsInput,
    state: SessionState,
) -> dict[str, Any]:
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str):
        return copy.deepcopy(dict(receipt))
    job = await services.resolve_job_async(job_id, state)
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


def _note_process_owned_job(receipt: dict[str, Any]) -> dict[str, Any]:
    """Say, on the receipt itself, that this job dies with the interpreter.

    ``Api.close()`` shuts the engine down, which cancels every job this process
    owns. A ``wait=False`` receipt otherwise looks exactly like a durable
    submission, and the loss is silent — the caller learns of it only when a
    later status read reports a cancellation nobody asked for. The contract
    states the rule; this is the same rule at the point of use.
    """
    if receipt.get("status") in experiments.TERMINAL_EXPERIMENT_STATUSES:
        return receipt
    observations = receipt.setdefault("observations", [])
    if isinstance(observations, list):
        observations.append(
            {
                "code": "process_owned_job",
                "kind": "lifecycle",
                "detail": (
                    "This job is owned by the current process and is cancelled when "
                    "the Api closes (including at the end of a 'with' block or when "
                    "the interpreter exits). Wait for it in this process, or submit "
                    "work that must outlive the interpreter through a long-lived "
                    "server."
                ),
            }
        )
    return receipt


def _note_detached_owner(
    receipt: dict[str, Any],
    *,
    owner_pid: int,
    log_file: Path,
) -> dict[str, Any]:
    """Say who owns this job now, and where that process writes.

    The receipt was rendered by the detached owner, so it carries that
    process's ``process_owned_job`` note — true there and misleading here,
    where "the current process" is the caller's and does not own the job. One
    fact replaces the other rather than sitting beside it.

    A detached call carrying a ``request_id`` that already ran to completion
    replays that job, and the pid on its record is a process that exited when
    the job did. Telling the caller that pid is supervising the job and can be
    cancelled is an instruction it may act on — worse still if the pid has
    since been recycled — so the terminal case says what is actually true.
    """
    observations = receipt.get("observations")
    if not isinstance(observations, list):
        observations = []
        receipt["observations"] = observations
    observations[:] = [
        item
        for item in observations
        if not (isinstance(item, Mapping) and item.get("code") == "process_owned_job")
    ]
    if receipt.get("status") in experiments.TERMINAL_EXPERIMENT_STATUSES:
        detail = (
            f"This job is already terminal. Pid {owner_pid} is the process its "
            "record names as having run it; nothing is supervising it now and "
            "there is nothing to cancel. Read its results with "
            "jobs(action='status'|'runs') or analyze_results by job_id. The "
            "process this call spawned to look the job up wrote its console "
            f"output to {log_file}."
        )
    else:
        detail = (
            f"This job is owned by a detached process (pid {owner_pid}) that "
            "supervises it until it is terminal. The current process does not "
            "own it, so closing the Api or exiting the interpreter will not "
            "cancel it. Read it back with jobs(action='status'|'wait') by "
            "job_id, and stop it with jobs(action='cancel') and this receipt's "
            f"control_token. The owner's console output is at {log_file}."
        )
    observations.append(
        {
            "code": "detached_owner",
            "kind": "lifecycle",
            "detail": detail,
            "evidence": {"owner_pid": owner_pid, "log_file": str(log_file)},
        }
    )
    return receipt


def _submitted_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """The same arguments with the wire dwell removed.

    The API always submits with ``execution.wait_s=0`` — the dwell is a
    presentation constant this interface rejects as caller input — and the
    detached owner submits the same way, so the two produce the same
    idempotency fingerprint for the same call.
    """
    execution = arguments.get("execution")
    if execution is None:
        arguments["execution"] = {"wait_s": 0}
    elif isinstance(execution, Mapping):
        arguments["execution"] = {**execution, "wait_s": 0}
    return arguments


async def _record_owner_pid(receipt: Mapping[str, Any], state: SessionState) -> int | None:
    """The pid recorded on the job's own record, or None if it names no job.

    Read from the record rather than taken from the process just spawned: an
    idempotent replay hands back a job somebody else already owns, and naming
    the wrong process as its owner is worse than naming none.
    """
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str):
        return None
    job = await services.resolve_job_async(job_id, state)
    owner_pid = getattr(job, "owner_pid", None)
    return owner_pid if isinstance(owner_pid, int) and owner_pid > 0 else None


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
            # The set is immutable and the previous drive holds it, so a
            # continuation resumes without re-reading it from disk.
            loaded=drives[-1].item if drives else None,
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
    """Distinct values in first-appearance order, sharing the inputs' rows."""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _merge_inspect_item(
    accumulated: dict[str, Any] | None,
    page_item: Mapping[str, Any],
    original_index: int,
) -> dict[str, Any]:
    # Every page arrives already detached by _unwrap and is read only here, so
    # the merge concatenates row references instead of copying the accumulated
    # rows again on each page.
    current = dict(page_item)
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
            new_data[key] = [*old_rows, *(new_rows if isinstance(new_rows, list) else [])]
        elif isinstance(old_value, list) and isinstance(new_data.get(key), list):
            new_data[key] = _dedupe([*old_value, *new_data[key]])

    for collection, metadata in page["collections"].items():
        if not isinstance(metadata, dict):
            continue
        rows = new_data.get(collection)
        returned = len(rows) if isinstance(rows, list) else int(metadata.get("returned", 0))
        metadata["returned"] = returned
        metadata["truncated"] = returned < int(metadata.get("total", returned))
        counters = inspect_tools.COLLECTION_COUNTERS.get(collection)
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
    # ``queries`` is SkipValidation, so its items are whatever the caller
    # passed — dicts on MCP and the Python API, models when Python code builds them.
    # Dumping the whole request makes pydantic serialize each dict against the
    # union member it was declared as, which warns to stderr on every
    # successful call; serialize the models and take the dicts as they are.
    raw_queries = [
        query.model_dump(mode="json") if isinstance(query, BaseModel) else dict(query)
        for query in request.queries
    ]
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


async def _collect_jobs(
    request: jobs.JobsInput,
    state: SessionState,
) -> dict[str, Any]:
    """Collect one jobs action: the whole circuit list, or a complete receipt.

    One evaluation, rendered complete. The MCP renders the same
    evaluation as a page — MCP and the Python API differ by that presentation argument and
    by nothing else, so neither can report a job the other did not read.
    """
    evaluation = await jobs.evaluate_jobs(request, state)
    data = jobs.complete_jobs_data(evaluation)
    if evaluation.is_error:
        message = _payload_message(data) or "The engine returned a call-level error"
        raise ApiCallError(message, payload=data)
    return data


async def _collect_edit_schematic(
    request: schematic_edit.EditSchematicInput,
    state: SessionState,
) -> dict[str, Any]:
    result = await schematic_edit.evaluate_edit_schematic(request, state)
    return schematic_edit.complete_edit_schematic_data(result, request)


async def _collect_verify_circuit(
    request: verify.VerifyCircuitInput,
    state: SessionState,
) -> dict[str, Any]:
    result = await verify.evaluate_verify_circuit(request, state)
    payload = copy.deepcopy(result.data)
    if result.is_error:
        rendered = verify.render_verify_circuit(result)
        raise ApiCallError(_message_for_error(payload, rendered), payload=payload)
    return payload


def _resolve_cancel(policy: _CancelPolicy, request: BaseModel) -> bool:
    return policy(request) if callable(policy) else policy


class ApiMethodsMixin(ABC):
    """Public consolidated methods mixed into :class:`ltspice_mcp.api.Api`."""

    _state: SessionState
    #: The constructor arguments a detached owner is given so it opens the same
    #: engine this session opened.
    _boot: _detach.DetachedBoot
    #: Owners spawned by this session. Held only so the finished ones can be
    #: reaped; a running one is never waited on.
    _detached_children: list[subprocess.Popen[bytes]]

    @staticmethod
    def reference(op: str | None = None) -> str:
        """Print the argument catalogue: the six-op index, or one op's tree.

        ``reference()`` lists the operations; ``reference('edit_schematic')``
        gives that operation's whole argument tree — every field with its type,
        default, enum members and union branches written out, nested models
        flattened onto dotted paths, and a worked example. Generated from the
        models the call validates against, so it says what will be accepted.

        A static method deliberately: reading the catalogue must not require an
        engine session, so ``Api.reference('inspect')`` works before anything is
        opened and without taking this process's single session lease.
        """
        return _reference.reference(op)

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

    def _marshal(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        cancelable: bool = False,
        cancel_on_interrupt: bool = False,
        preserve_interrupt: bool = False,
    ) -> _T:
        """Marshal one call with relative paths anchored on the working dir.

        The Python API lets the caller name a working directory that is not their
        cwd, so ``Api(working_dir=D)`` plus a bare ``"opamp2.asc"`` — the idiom
        the contract documents — has to look in ``D``. Anchoring here rather
        than per path field means every op, and every path field a future op
        adds, inherits it: they all pass through this one call.
        """
        return self._call(
            _anchored_on(self._state.config.working_dir, coroutine),
            cancelable=cancelable,
            cancel_on_interrupt=cancel_on_interrupt,
            preserve_interrupt=preserve_interrupt,
        )

    def _dispatch(
        self,
        name: str,
        model: type[_ModelT],
        handler: Any,
        collector: _Collector,
        *,
        raw_page: bool,
        arguments: Mapping[str, Any],
        cancelable: _CancelPolicy = False,
        cancel_on_interrupt: _CancelPolicy = False,
    ) -> dict[str, Any]:
        """Validate one operation's arguments and marshal its chosen mode.

        ``raw_page`` selects a single handler page over the collected result and
        is also what admits the wire-only controls the automatic mode rejects.
        """
        self._check_process_and_thread()
        if not raw_page:
            _enforce_auto_door(arguments)
        request = _validate(name, model, arguments, self._state)
        coroutine = (
            _handler_page(handler, request, self._state)
            if raw_page
            else _through_auto_door(collector(request, self._state))
        )
        return self._marshal(
            coroutine,
            cancelable=_resolve_cancel(cancelable, request),
            cancel_on_interrupt=_resolve_cancel(cancel_on_interrupt, request),
        )

    def run_experiments(
        self,
        *,
        wait: bool = True,
        raw_page: bool = False,
        detach: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Submit an experiment and optionally wait for its complete receipt."""
        self._check_process_and_thread()
        if not isinstance(wait, bool):
            raise TypeError("wait must be a bool")
        if not isinstance(detach, bool):
            raise TypeError("detach must be a bool")
        if detach:
            if wait:
                raise ApiValidationError(
                    "detach=True requires wait=False: a detached job is owned by the "
                    "process spawned for it, so this process cannot wait on it as its "
                    "owner. Drop detach to run the job here, or pass wait=False and "
                    "follow the receipt with api.wait(job_id)."
                )
            if raw_page:
                raise ApiValidationError(
                    "detach=True cannot be combined with raw_page=True: raw_page returns "
                    "one handler page from a submission this process performs, and a "
                    "detached submission is performed by another process."
                )
            if not self._state.config.persist_jobs:
                raise ApiValidationError(
                    "detach=True needs persisted job records: a detached job is read "
                    "back from its record, and this session has [state] persist_jobs "
                    "off, so nothing the owner submits would be visible here."
                )
            return self._run_detached(arguments)
        if raw_page:
            request = _validate(
                "run_experiments", experiments.RunExperimentsInput, arguments, self._state
            )
            return self._marshal(
                _handler_page(experiments.handle_run_experiments, request, self._state)
            )

        _enforce_auto_door(arguments)
        submitted_arguments = _submitted_arguments(copy.deepcopy(arguments))
        request = _validate(
            "run_experiments",
            experiments.RunExperimentsInput,
            submitted_arguments,
            self._state,
        )
        receipt = self._marshal(
            _through_auto_door(
                _handler_page(experiments.handle_run_experiments, request, self._state)
            ),
            preserve_interrupt=True,
        )
        # The handler's own receipt already says whether anything is still
        # running, so the complete receipt is assembled once — for the value
        # actually returned — rather than before and after the wait.
        job_id = receipt.get("job_id")
        waited_job = (
            job_id
            if wait and isinstance(job_id, str) and receipt.get("outcome") == "in_progress"
            else None
        )
        try:
            if waited_job is not None:
                self.wait(waited_job)
            complete = self._marshal(
                _through_auto_door(_complete_run_receipt(receipt, request, self._state))
            )
            return complete if wait else _note_process_owned_job(complete)
        except KeyboardInterrupt as exc:
            raise ApiInterrupted(receipt=receipt, job_id=waited_job) from exc
        except Exception as exc:
            raise _collector_error(receipt, exc) from exc

    def _run_detached(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Submit through a process spawned for this one job, and return its receipt.

        The arguments are validated here so a malformed call still raises in
        the caller's own traceback, before any process exists — but nothing
        else about the experiment happens in this process. Staging and
        submission both belong to the owner, which is what makes the job record
        name a live owner from its first byte.
        """
        _enforce_auto_door(arguments)
        # The owner is given the call as the caller wrote it and applies the
        # same automatic-mode rules to it, dwell removal included — it is an
        # ordinary Api caller. Sending it a request that already carried
        # execution.wait_s would hand it a control its own door refuses.
        payload = _detach.request_arguments(arguments)
        request = _validate(
            "run_experiments",
            experiments.RunExperimentsInput,
            _submitted_arguments(copy.deepcopy(payload)),
            self._state,
        )
        handoff = _detach.submit(
            self._state,
            self._boot,
            payload,
            request.request_id,
            self._detached_children,
        )
        try:
            owner_pid = self._marshal(_record_owner_pid(handoff.receipt, self._state))
        except Exception as exc:
            raise _collector_error(handoff.receipt, exc) from exc
        return _note_detached_owner(
            handoff.receipt,
            owner_pid=owner_pid if owner_pid is not None else handoff.supervisor_pid,
            log_file=handoff.log_file,
        )

    def jobs(self, *, raw_page: bool = False, **arguments: Any) -> dict[str, Any]:
        """Control jobs, collecting list and receipt pages in automatic mode."""
        return self._dispatch(
            "jobs",
            jobs.JobsInput,
            jobs.handle_jobs,
            _collect_jobs,
            raw_page=raw_page,
            arguments=arguments,
            # A cancel is the one action an interrupt must not abandon halfway.
            # A collected list keeps paging until it is complete, so an
            # interrupt has to reach it the way it reaches a dwelling wait.
            cancelable=lambda request: request.action != "cancel",
            cancel_on_interrupt=lambda request: (
                request.action == "wait" or (not raw_page and request.action == "list")
            ),
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
            # Poll on the raw handler page: a dwell consumes nothing but
            # timed_out, and rendering a complete receipt per dwell would pay
            # for the whole wait in receipts nobody reads.
            try:
                page = self.jobs(
                    raw_page=True,
                    action="wait",
                    job_id=job_id,
                    timeout_s=dwell,
                )
                if not page.get("timed_out"):
                    return self.jobs(action="wait", job_id=job_id, timeout_s=0)
                if timeout is not None and time.monotonic() - started >= float(timeout):
                    result = self.jobs(action="wait", job_id=job_id, timeout_s=0)
                    result["timed_out"] = True
                    return result
            except ApiInterrupted:
                raise
            except KeyboardInterrupt as exc:
                raise ApiInterrupted(job_id=job_id) from exc

    def analyze_results(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Analyze results, driving bounded neutral evaluation to completion."""
        return self._dispatch(
            "analyze_results",
            analyze.AnalyzeResultsInput,
            analyze.handle_analyze_results,
            _collect_analysis,
            raw_page=raw_page,
            arguments=arguments,
            cancelable=True,
            cancel_on_interrupt=True,
        )

    def inspect(self, *, raw_page: bool = False, **arguments: Any) -> dict[str, Any]:
        """Run batched read queries, collecting every per-query cursor."""
        return self._dispatch(
            "inspect",
            inspect_tools.InspectInput,
            inspect_tools.handle_inspect,
            _collect_inspect,
            raw_page=raw_page,
            arguments=arguments,
            cancelable=True,
            cancel_on_interrupt=True,
        )

    def edit_schematic(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Apply one schematic transaction and return its complete in-memory views."""
        # Never cancelable: the transaction owns the commit protocol, and the
        # sheet is either fully written or not written at all.
        return self._dispatch(
            "edit_schematic",
            schematic_edit.EditSchematicInput,
            schematic_edit.handle_edit_schematic,
            _collect_edit_schematic,
            raw_page=raw_page,
            arguments=arguments,
        )

    def verify_circuit(
        self,
        *,
        raw_page: bool = False,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Verify a circuit, returning every finding from the neutral evaluator."""
        return self._dispatch(
            "verify_circuit",
            verify.VerifyCircuitInput,
            verify.handle_verify_circuit,
            _collect_verify_circuit,
            raw_page=raw_page,
            arguments=arguments,
            cancelable=not raw_page,
            cancel_on_interrupt=not raw_page,
        )

    def load_raw(
        self,
        *,
        raw_path: str | os.PathLike[str] | None = None,
        job_id: str | None = None,
        run_index: int = 0,
        case_id: str | None = None,
    ) -> RawResult:
        """Load one raw result through the bounded parser and return a safe wrapper."""
        self._check_process_and_thread()
        if (raw_path is None) == (job_id is None):
            raise TypeError("Pass exactly one of raw_path or job_id")
        if raw_path is not None and (run_index != 0 or case_id is not None):
            raise TypeError("run_index and case_id are only valid with job_id")
        return self._marshal(
            load_raw_result(
                state=self._state,
                raw_path=None if raw_path is None else os.fspath(raw_path),
                job_id=job_id,
                run_index=run_index,
                case_id=case_id,
            ),
            cancelable=True,
            cancel_on_interrupt=True,
        )

    def measurements(
        self,
        *,
        job_id: str,
        run_index: int = 0,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """Return parsed ``.meas`` data for one experiment case."""
        self._check_process_and_thread()
        return self._marshal(
            load_measurement_results(
                state=self._state,
                job_id=job_id,
                run_index=run_index,
                case_id=case_id,
            ),
            cancelable=True,
            cancel_on_interrupt=True,
        )


# The six methods carry their operation's catalogue entry as their docstring —
# from the same renderer api.reference uses, so the two can never disagree.
# Installed LAZILY: rendering needs the live tool models, whose import is
# exactly what the engine boot defers, so installation rides the first event
# that pays that import anyway — the first deferred tool-module resolution
# (any operation call) or the first catalogue read (reference()). Until one
# of those happens, help() on a method shows only its signature.
_method_docs_installed = False


def ensure_method_docs() -> None:
    """Install the catalogue docstrings once (idempotent, import-triggered)."""
    global _method_docs_installed
    if not _method_docs_installed:
        _method_docs_installed = True
        _reference.install_method_docs(ApiMethodsMixin)
