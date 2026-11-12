"""Structured events for experiment jobs and tool calls.

Emits machine-parseable log records on job state transitions — submit,
start, completion, failure, cancellation, interruption, interrupted recovery —
and one record per tool call (:func:`emit_tool_event`). Logs go through a
dedicated ``ltspice_mcp.events`` logger so operators can route them to a
different sink from the usual debug/info stream.

Each job event carries:
    ts            ISO-8601 timestamp (UTC)
    event         lifecycle state: submitted | started | analyzing
                  | completed | completed_with_failures | failed
                  | cancelled | interrupted | interrupted_recovered
    kind          'experiment' — the one job kind that runs here
    job_id        job identifier
    sources       circuit paths the job runs over
    duration_s    wall-clock seconds from started_at to now (or None
                  when the event precedes ``started_at``)
    extra keys    anything passed via kwargs (e.g. error, total_cases,
                  recovered_as)

A tool call's event (:func:`emit_tool_event` over MCP,
:func:`emit_api_event` through the Python API), a resource read's and a prompt
get's (:func:`emit_read_event`) are described in ``docs/design/mcp_surface.md``
("Server logging"). They carry names only, never a value the caller sent:
logs get shared.

Payloads are attached to log records via the ``extra`` dict so a
structured-log shipper (python-json-logger etc.) can pick them up
without parsing the message string.

Stderr is the one channel these records travel on, so installing it lives
here too: :func:`configure_stderr_logging` is what the server's lifespan and
the detached owner both call, and it is the only place that decides the
format an operator reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import functools
import logging
import sys
import types as pytypes
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    ParamSpec,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, SkipValidation, TypeAdapter

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES

if TYPE_CHECKING:
    from mcp import types

logger = logging.getLogger("ltspice_mcp.events")

#: The stderr format every ltspice-mcp process writes.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_stderr_logging(level: str) -> None:
    """Install this process's stderr logging at ``level``.

    An unrecognized level falls back to INFO rather than raising: this runs at
    startup, before there is anywhere to report a bad value, and refusing to
    boot over a log level would be worse than logging more than asked.

    ``force=True``, so a process may call this again once it knows more —
    which is what the detached owner does after its config is loaded.
    """
    from ltspice_mcp.config import VALID_LOG_LEVELS

    named = level.upper()
    if named not in VALID_LOG_LEVELS:
        named = "INFO"
    logging.basicConfig(
        level=getattr(logging, named),
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


JobEvent = Literal[
    "submitted",
    "started",
    "analyzing",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
    "interrupted",
    "interrupted_recovered",
]


def _duration_seconds(started_at: datetime | None) -> float | None:
    """Wall-clock seconds since ``started_at`` in UTC, or None."""
    if started_at is None:
        return None
    # Both sides are tz-aware via ``lib.now()`` so subtraction yields a
    # correct timedelta regardless of the zone choice.
    current = now()
    # Guard against clock skew / reordered events.
    delta = (current - started_at).total_seconds()
    return max(delta, 0.0)


def emit_job_event(
    event: JobEvent,
    job: ExperimentJob,
    **extra: Any,
) -> None:
    """Emit a structured lifecycle event for ``job``.

    Any additional keyword args are merged into the event payload.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        # Every job this server runs is an experiment.
        "kind": "experiment",
        "job_id": job.job_id,
        "duration_s": _duration_seconds(getattr(job, "started_at", None)),
    }
    payload["sources"] = [str(source.path) for source in job.sources]
    payload.update(extra)

    # Human-readable summary in the message, structured dict in extra.
    suffix = ""
    if payload.get("duration_s") is not None and event in TERMINAL_STATUSES:
        suffix = f" after {payload['duration_s']:.2f}s"
    logger.info(
        "experiment.%s job=%s%s",
        event,
        job.job_id,
        suffix,
        extra={"ltspice_event": payload},
    )


_P = ParamSpec("_P")


def _never_raises(emit: Callable[_P, None]) -> Callable[_P, None]:
    """Run an emitter so that a failure building its record is logged and
    swallowed: an event describes a call, and must never replace the call's
    own result or its own exception."""

    @functools.wraps(emit)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> None:
        try:
            emit(*args, **kwargs)
        except Exception:
            logger.debug("could not build the event of %s", emit.__name__, exc_info=True)

    return guarded


#: Which way a call reached the tool handlers: an MCP ``tools/call``, or an
#: operation of the Python API (``ltspice_mcp.api``, which ``run_code`` uses).
Door = Literal["mcp", "python_api"]

# The request one call validated, held for the call's event. ``capturing_request``
# opens a holder; the first ``note_request`` inside it fills it. The holder is a
# list so a note made in a context copied from the opener's (a task the call
# starts) still reaches it.
_REQUEST: contextvars.ContextVar[list[BaseModel] | None] = contextvars.ContextVar(
    "ltspice_mcp_event_request", default=None
)


def events_enabled() -> bool:
    """Whether a call's event would be written: what building one costs is
    paid only then."""
    return logger.isEnabledFor(logging.INFO)


@contextlib.contextmanager
def capturing_request() -> Iterator[list[BaseModel]]:
    """Hold the request the call inside validates, for its event."""
    holder: list[BaseModel] = []
    token = _REQUEST.set(holder)
    try:
        yield holder
    finally:
        _REQUEST.reset(token)


def note_request(request: BaseModel) -> None:
    """Record ``request`` as the validated arguments of the call being
    captured, unless one is already recorded: a handler that delegates to
    another validates again, and the event describes what the caller sent."""
    holder = _REQUEST.get()
    if holder is not None and not holder:
        holder.append(request)


#: The marker ``SkipValidation[X]`` puts in ``X``'s metadata (a class at run
#: time, an alias to the type checker).
_SKIP_VALIDATION = cast(type, SkipValidation)

_ADAPTERS: dict[int, tuple[Any, TypeAdapter[Any]]] = {}


def _adapter(annotation: Any) -> TypeAdapter[Any]:
    """A validator for an annotation a model keeps unvalidated, built once."""
    held = _ADAPTERS.get(id(annotation))
    if held is None or held[0] is not annotation:
        held = (annotation, TypeAdapter(annotation))
        _ADAPTERS[id(annotation)] = held
    return held[1]


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """``annotation`` without ``SkipValidation``, and whether it carried it.

    A model leaves a ``SkipValidation`` value as the caller sent it (the
    ``analyze_results`` recipes, which its handler validates one by one), so
    the walk validates it to read its fields."""
    if get_origin(annotation) is Annotated:
        inner, *metadata = get_args(annotation)
        kept = [m for m in metadata if not isinstance(m, _SKIP_VALIDATION)]
        if len(kept) != len(metadata):
            return (Annotated[inner, *kept] if kept else inner), True
    return annotation, False


def _vocabulary(annotation: Any) -> frozenset[str]:
    """The strings ``annotation`` admits by name: its ``Literal`` and enum
    members, at any depth. A value outside this set is the caller's own."""
    found: set[str] = set()
    pending = [annotation]
    while pending:
        current = pending.pop()
        origin = get_origin(current)
        if origin is Literal:
            found.update(a for a in get_args(current) if isinstance(a, str))
        elif isinstance(current, type) and issubclass(current, enum.Enum):
            found.update(m.value for m in current if isinstance(m.value, str))
        elif origin is not None:
            pending.extend(a for a in get_args(current) if not isinstance(a, (str, int)))
    return frozenset(found)


def _items_annotation(annotation: Any) -> Any:
    """The annotation of one item of a list, tuple or mapping ``annotation``
    admits, as a union when it admits several."""
    options: list[Any] = []
    pending = [annotation]
    while pending:
        current = pending.pop()
        origin = get_origin(current)
        if origin is Annotated:
            pending.append(get_args(current)[0])
        elif origin in (Union, pytypes.UnionType):
            pending.extend(get_args(current))
        elif origin in (list, tuple, set, frozenset) or origin is Mapping or origin is dict:
            args = [a for a in get_args(current) if a is not Ellipsis]
            if args:
                options.append(args[-1])
    if not options:
        return Any
    return options[0] if len(options) == 1 else Union[tuple(options)]  # noqa: UP007


def _is_default(model: BaseModel, name: str, value: Any) -> bool:
    field = type(model).model_fields[name]
    if field.is_required():
        return False
    try:
        return bool(value == field.get_default(call_default_factory=True))
    except Exception:
        return False


def _declared_discriminator(annotation: Any) -> str | None:
    """The field a union ``annotation`` declares as its discriminator
    (``Field(discriminator="kind")``), read from its own metadata: the one
    answer pydantic itself validates the union by."""
    if get_origin(annotation) is not Annotated:
        return None
    for metadata in get_args(annotation)[1:]:
        named = getattr(metadata, "discriminator", None)
        if isinstance(named, str):
            return named
    return None


def _member_tag(model: type[BaseModel], discriminator: str) -> tuple[str, str] | None:
    """The wire name and word of ``model``'s ``discriminator`` field, which
    tag it as one member of its union."""
    for name, field in model.model_fields.items():
        if discriminator not in (name, field.alias):
            continue
        words = get_args(field.annotation) if get_origin(field.annotation) is Literal else ()
        if len(words) == 1 and isinstance(words[0], str):
            return field.alias or name, words[0]
        return None
    return None


def _members(annotation: Any) -> list[type[BaseModel]]:
    """The models ``annotation`` admits, at any depth of unions and metadata."""
    found: list[type[BaseModel]] = []
    pending = [annotation]
    while pending:
        current = pending.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
        elif get_origin(current) in (Annotated, Union, pytypes.UnionType):
            pending.extend(get_args(current))
    return found


def _refused_item(
    value: Any,
    annotation: Any,
    discriminator: str | None,
    path: str,
    fields: set[str],
    choices: set[str],
) -> None:
    """Name what a refused item of a tagged union tried: its tag, when it is
    one a member declares, and those of its keys that member declares."""
    if discriminator is None or not isinstance(value, Mapping):
        return
    for member in _members(annotation):
        tag = _member_tag(member, discriminator)
        if tag is None or value.get(tag[0]) != tag[1]:
            continue
        choices.add(f"{path}.{tag[0]}={tag[1]}")
        declared = {field.alias or name for name, field in member.model_fields.items()}
        prefix = f"{path}[{tag[0]}={tag[1]}]"
        fields.update(f"{prefix}.{key}" for key in value if key in declared and key != tag[0])
        return


def _walk(
    value: Any,
    annotation: Any,
    path: str,
    fields: set[str],
    choices: set[str],
    discriminator: str | None = None,
) -> None:
    discriminator = _declared_discriminator(annotation) or discriminator
    annotation, skipped = _unwrap(annotation)
    if skipped and not isinstance(value, BaseModel):
        try:
            value = _adapter(annotation).validate_python(value)
        except Exception:
            # The handler refuses this item and answers the others. What the
            # item tried is still a use: its tag and the names its member
            # declares, never a key the caller made up.
            _refused_item(value, annotation, discriminator, path, fields, choices)
            return
    if isinstance(value, BaseModel):
        model_fields = type(value).model_fields
        tag = _member_tag(type(value), discriminator) if path and discriminator else None
        prefix = path
        if tag is not None:
            # One item of a tagged union: its fields are named under its tag,
            # so a call with several items still says which kind set what.
            choices.add(f"{path}.{tag[0]}={tag[1]}")
            prefix = f"{path}[{tag[0]}={tag[1]}]"
        for name in value.model_fields_set:
            field = model_fields.get(name)
            if field is None or (tag is not None and (field.alias or name) == tag[0]):
                continue
            child = getattr(value, name, None)
            if _is_default(value, name, child):
                continue
            wire = field.alias or name
            child_path = f"{prefix}.{wire}" if prefix else wire
            fields.add(child_path)
            declared = field.discriminator if isinstance(field.discriminator, str) else None
            _walk(child, field.annotation, child_path, fields, choices, declared)
    elif isinstance(value, (list, tuple, set, frozenset)):
        item = _items_annotation(annotation)
        for element in value:
            _walk(element, item, path, fields, choices, discriminator)
    elif isinstance(value, Mapping):
        # A mapping's keys are the caller's (parameter names, labels), so only
        # its values are walked, as a list's items are.
        item = _items_annotation(annotation)
        for element in value.values():
            _walk(element, item, path, fields, choices, discriminator)
    else:
        word = value.value if isinstance(value, enum.Enum) else value
        if isinstance(word, str) and path and word in _vocabulary(annotation):
            choices.add(f"{path}={word}")


def request_names(request: BaseModel) -> tuple[list[str], list[str]]:
    """The argument names a validated ``request`` set, and the words of the
    server's own vocabulary its values chose.

    A name is a field path: nested fields count, list positions and mapping
    keys are dropped, and each path appears once. A field the caller left out,
    or sent at its default, is not set. The fields of an item tagged as one
    member of a discriminated union are named under its declared discriminator
    (``recipes[metric=value].spec.allow_incomplete``,
    ``queries[kind=hierarchy].cursor``). A choice is ``path=word`` for a value
    a ``Literal`` or enum of the field's own type names
    (``queries.kind=simulator_docs``, ``recipes.metric=plot``,
    ``ops.op=add_component``, ``action=wait``); no other part of a value is
    read, so no path, signal, number, expression or text can reach the record.
    """
    fields: set[str] = set()
    choices: set[str] = set()
    _walk(request, type(request), "", fields, choices)
    return sorted(fields), sorted(choices)


@_never_raises
def emit_tool_event(
    tool: str | None,
    duration_s: float,
    ended: types.CallToolResult | BaseException,
    request: BaseModel | None = None,
) -> None:
    """Emit one event for an MCP call of ``tool`` that ``ended`` with its
    result or by raising, a cancellation included. ``tool`` is None for a
    name this server serves no tool by, which is the caller's text.

    ``request`` is the call's validated arguments, None when they were refused
    before validation finished; the record carries their names and vocabulary
    choices (:func:`request_names`), never a value.

    Measuring a reply serializes it again, so nothing is measured unless the
    record would be emitted.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    code = size = None
    if isinstance(ended, BaseException):
        outcome = "cancelled" if isinstance(ended, asyncio.CancelledError) else "raised"
    else:
        outcome = "error" if ended.is_error else "ok"
        if ended.is_error:
            code = (ended.structured_content or {}).get("code")
        size = len(ended.model_dump_json(by_alias=True, exclude_none=True))
    _emit_call("mcp", tool, duration_s, outcome, code, size, request)


@_never_raises
def emit_api_event(
    operation: str,
    duration_s: float,
    outcome: Literal["ok", "error", "cancelled", "raised"],
    code: str | None,
    request: BaseModel | None,
    controls: Iterable[str],
) -> None:
    """Emit one event for a Python API operation call: a tool call's record,
    through the door ``python_api``.

    There is no wire reply, so ``reply_chars`` is None. ``controls`` are the
    API's own options the call set (``raw_page``, ``wait``, ``detach``).
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    _emit_call(
        "python_api", operation, duration_s, outcome, code, None, request, sorted(set(controls))
    )


def _emit_call(
    door: Door,
    tool: str | None,
    duration_s: float,
    outcome: str,
    code: Any,
    size: int | None,
    request: BaseModel | None,
    controls: list[str] | None = None,
) -> None:
    fields = choices = None
    if request is not None:
        fields, choices = request_names(request)
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": "tool_call",
        "door": door,
        "tool": tool,
        "duration_s": duration_s,
        "outcome": outcome,
        "code": code,
        "reply_chars": size,
        "fields": fields,
        "choices": choices,
    }
    if controls is not None:
        payload["controls"] = controls
    logger.info(
        "tool.%s %s after %.2fs%s",
        tool,
        outcome,
        duration_s,
        "" if size is None else f", {size} chars",
        extra={"ltspice_event": payload},
    )


@_never_raises
def emit_read_event(
    event: Literal["resource_read", "prompt_get"],
    name: str | None,
    duration_s: float,
    outcome: str,
    code: Any = None,
) -> None:
    """Emit one event for a resource read or a prompt get.

    ``name`` is the URI template the read matched (``spice://results/{job_id}/
    signals``) or the prompt's name, None when the request named none this
    server serves: never the concrete URI or a prompt argument, which carry
    the caller's paths and signals.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    key = "template" if event == "resource_read" else "prompt"
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        key: name,
        "duration_s": duration_s,
        "outcome": outcome,
        "code": code,
    }
    logger.info(
        "%s %s %s after %.2fs",
        event,
        name,
        outcome,
        duration_s,
        extra={"ltspice_event": payload},
    )
