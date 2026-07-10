"""Shared utilities for tool handlers."""

import asyncio
import contextlib
import copy
import json
import logging
import re
import types as _stdlib_types
import typing
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from mcp import types
from pydantic import BaseModel, ConfigDict

from ltspice_mcp.errors import NetlistError, SimulationError
from ltspice_mcp.lib.filelock import DEFAULT_TIMEOUT, file_lock
from ltspice_mcp.lib.job_store import SIDECAR_DIRNAME
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.runner_base import LOGOPINFO_MARKER
from ltspice_mcp.lib.simulator import no_simulator_message
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response helpers — standardize tool output format
#
# All helpers return types.CallToolResult, the MCP protocol's canonical
# response type.  text_response() returns text-only (for confirmations).
# format_response() returns both human-readable text content AND structured
# data via structuredContent (for data-returning tools).
# ---------------------------------------------------------------------------


def text_response(text: str) -> types.CallToolResult:
    """Return a text-only CallToolResult (confirmations, simple messages)."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
    )


def json_response(data: Any) -> types.CallToolResult:
    """Return data as JSON text + structuredContent."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=2))],
        structuredContent=data,
    )


def format_response(
    text: str,
    data: Mapping[str, Any],
    fmt: str | None = None,
) -> types.CallToolResult:
    """Return a CallToolResult with text content and structuredContent.

    Always populates structuredContent for programmatic access. Accepts any
    mapping so TypedDict return values from lib functions pass through
    without cast/copy. The format param controls the text representation:
    - "json": text is JSON-formatted (for clients that parse text)
    - "text" or None: text is human-readable (default)

    Self-sufficiency contract: structured-aware clients (Claude Code included)
    render ONLY structuredContent when it is present and drop the text channel
    entirely, so ``data`` must carry everything the caller needs to act on.
    Any caller-guidance composed into ``text`` (hints, referrals, recovery
    steps, caveats) must be mirrored into ``data`` — conventionally an
    optional ``hint`` key declared in the tool's output_schema. The text
    channel is presentation only.
    """
    # MCP SDK's CallToolResult wants a plain dict for structuredContent;
    # TypedDicts ARE plain dicts at runtime, but wrap defensively so
    # non-dict mappings (rare, but cheap to support) also work.
    payload: dict[str, Any] = dict(data) if not isinstance(data, dict) else data
    if fmt == "json":
        return json_response(payload)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=payload,
    )


# ---------------------------------------------------------------------------
# Shared tool schema fragments and annotations
# ---------------------------------------------------------------------------

# Client-visible description of the shared ``format`` input param — one string,
# reused by every tool input model that exposes the param.
FORMAT_DESCRIPTION = (
    "Response format: 'json' for structured data, 'text' for human-readable "
    "(default; both carry the same structured content)"
)

# Output-schema fragment for the optional ``hint`` key: caller guidance
# mirrored from the text channel (see format_response's self-sufficiency
# contract). Sites needing a custom description inline their own dict.
HINT_SCHEMA: dict[str, str] = {"type": "string"}

# Free-text measurement caveats (see the observations-vs-warnings doctrine in
# lib/result_observations.py).
WARNINGS_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

# Fuzzy library matches for unresolved model/subcircuit references, keyed by
# the missing ref: ``{ref: [{name, score, source_path}, ...]}`` (produced by
# services.suggestions_from_errors / attach_suggestions_to_failure).
SUGGESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
                "source_path": {"type": "string"},
            },
        },
    },
}

PAGINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total": {"type": "integer"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
        "has_more": {"type": "boolean"},
        "next_offset": {"type": ["integer", "null"]},
    },
}

PIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "dir": {"type": "string"},
        "order": {"type": "integer"},
    },
}

BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
}

# Structured advisories emitted by mutating .asc handlers after a successful
# op. ``message`` is always present and human-readable; the other keys
# depend on ``kind``. New kinds extend ``VALIDATION_WARNING_KINDS`` and the
# schema enum together so producers and consumers stay in lockstep.
VALIDATION_WARNING_KINDS: tuple[str, ...] = (
    "floating_pin",
    "duplicate_wire",
    "dangling_label",
)

VALIDATION_WARNINGS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(VALIDATION_WARNING_KINDS)},
            "message": {"type": "string"},
            "ref": {"type": "string"},
            "pin": {"type": "string"},
            "label": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "from": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "to": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "count": {"type": "integer"},
        },
        "required": ["kind", "message"],
    },
}

# Structured .MEAS parse errors surfaced from extract_log_diagnostics.
# Each entry is the offending directive plus an optional fix suggestion
# pulled from the spice_validator blocklist.
MEAS_ERRORS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "directive": {"type": "string"},
            "raw_block": {"type": "string"},
            "suggestion": {"type": ["string", "null"]},
        },
    },
}

# Surfaced result observations — see lib/result_observations.py and the
# "Result-trust: surface, don't judge" in CLAUDE.md. A "surfacer"
# layer: facts lifted into view for the consuming agent to judge, never a trust
# verdict. ``severity`` is present only on ``relay`` items (the simulator's own
# classification); ``value``/``reconciliation``/``coverage`` items omit it.
OBSERVATIONS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": ["relay", "reconciliation", "value", "coverage"],
            },
            "detail": {"type": "string"},
            "severity": {"type": "string"},
            "evidence": {"type": "object"},
        },
        "required": ["code", "kind", "detail"],
    },
}

# Parsed .MEAS results, keyed by measurement name. Shared by run_simulation,
# check_job, and simulation_summary so the WHEN/AT field semantics are
# described identically everywhere. For a plain value measurement ``values``
# holds the scalar(s); for a WHEN/AT point measurement ``values`` is the
# constant trigger LEVEL and the crossing time/frequency lives in ``at`` —
# the descriptions below are the only place that distinction is self-evident
# to a client reading the schema.
MEASUREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": ["number", "null"]},
                "description": (
                    "Per-.step scalar(s). For a WHEN/AT measurement this is the "
                    "constant trigger LEVEL, not the crossing point — read 'at' "
                    "for the crossing time/frequency."
                ),
            },
            # Scalar when the bound is constant across .step iterations; list
            # (one entry per step) when it varies (e.g. TRIG/TARG marker times).
            "range_from": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
            },
            "range_to": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
            },
            "at": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
                "description": (
                    "Crossing time (.tran) or frequency (.ac) for a WHEN/AT point "
                    "measurement — THIS is the answer for a WHEN rise-time/crossing "
                    "query; 'values' holds the constant level. Null for plain "
                    "value measurements."
                ),
            },
        },
        "required": ["values"],
    },
}


def format_meas_errors(meas_errors: list[dict[str, Any]]) -> list[str]:
    """Render structured .MEAS errors for the text-format response.

    Returns the lines (no trailing blank); callers append to their own
    line list. Empty input returns an empty list so callers don't need
    to guard.
    """
    if not meas_errors:
        return []
    lines = [f".MEAS errors ({len(meas_errors)}):"]
    for me in meas_errors:
        lines.append(f"  Directive: {me['directive']}")
        if me.get("suggestion"):
            lines.append(f"    Suggestion: {me['suggestion']}")
    return lines


def format_observations(observations: list[dict[str, Any]]) -> list[str]:
    """Render surfaced result observations for a text-format response.

    Relay observations are omitted — they already print in the Errors section —
    so this shows the new facts (reconciliation/value/coverage). Returns the
    lines (no trailing blank); empty / all-relay input returns ``[]``. The full
    list (including relay) always rides in structuredContent.
    """
    surfaced = [o for o in observations if o.get("kind") != "relay"]
    if not surfaced:
        return []
    lines = ["Observations (facts to weigh, not a verdict):"]
    lines.extend(f"  [{o.get('kind')}] {o.get('detail')}" for o in surfaced)
    return lines


RO_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class StrictModel(BaseModel):
    """Shared Pydantic config for all strict models (tool inputs and nested schemas)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ToolInput(StrictModel):
    """Base for top-level tool input models registered via @registry.tool(input_model=...)."""

    pass


@dataclass(frozen=True)
class RegisteredTool:
    """Tool registration metadata used by the dispatch layer."""

    definition: types.Tool
    handler: Callable
    input_model: type[ToolInput] | None
    profiles: frozenset[str]


def _strip_titles(node: Any) -> Any:
    """Recursively remove Pydantic title metadata from a JSON schema node."""
    if isinstance(node, dict):
        node = {k: _strip_titles(v) for k, v in node.items() if k != "title"}
        return node
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


def _inline_json_schema(node: Any, defs: dict[str, Any]) -> Any:
    """Inline ``$defs`` references in a Pydantic-generated schema."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            resolved = copy.deepcopy(defs[name])
            return _inline_json_schema(resolved, defs)
        return {key: _inline_json_schema(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_json_schema(item, defs) for item in node]
    return node


def _build_input_schema(input_model: type[ToolInput]) -> dict[str, Any]:
    """Generate a cleaned MCP-ready JSON schema from a Pydantic model."""
    schema = input_model.model_json_schema()
    defs = schema.pop("$defs", {})
    schema = _inline_json_schema(schema, defs)
    return _strip_titles(schema)


# ---------------------------------------------------------------------------
# TypedDict → JSON Schema generator
# ---------------------------------------------------------------------------


_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and typing.is_typeddict(tp)


def _jsontype_from_union(args: tuple[Any, ...]) -> dict[str, Any]:
    """Handle ``X | None`` and ``X | Y | None`` unions.

    ``X | None`` becomes ``{"type": ["X", "null"]}`` when X is a single
    primitive — the common case for ``float | None`` fields. Mixed unions
    with complex members fall back to ``anyOf``.
    """
    non_none = [a for a in args if a is not type(None)]
    has_none = len(non_none) != len(args)
    if len(non_none) == 1:
        inner = _schema_for_type(non_none[0])
        if has_none and "type" in inner and isinstance(inner["type"], str):
            type_val = inner["type"]
            return {**inner, "type": [type_val, "null"]}
        if has_none:
            # Complex inner (nested object/array) — use anyOf with null.
            return {"anyOf": [inner, {"type": "null"}]}
        return inner
    variants = [_schema_for_type(a) for a in non_none]
    if has_none:
        variants.append({"type": "null"})
    return {"anyOf": variants}


def _is_union(tp: Any) -> bool:
    """True for both ``typing.Union[X, Y]`` and ``X | Y`` syntax."""
    if get_origin(tp) is Union:
        return True
    # Python 3.10+: `X | Y` has origin == types.UnionType (the class).
    return get_origin(tp) is _stdlib_types.UnionType


def _schema_for_type(tp: Any) -> dict[str, Any]:
    """Return a JSON Schema fragment for a type annotation."""
    if tp is Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    if tp in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[tp]}
    if _is_typeddict(tp):
        return schema_from_typeddict(tp)

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        return {"enum": list(args)}
    if _is_union(tp):
        return _jsontype_from_union(args)
    if origin in (list, tuple):
        # A fixed, heterogeneous tuple (``tuple[int, str]``) has no single
        # ``items`` schema; rendering ``args[0]`` only would SILENTLY drop the
        # rest, so refuse it loudly (use a TypedDict or list[...], or add
        # prefixItems support) rather than emit a schema that lies about the
        # shape. A list, a ``tuple[X, ...]``, and a single-type/empty tuple all
        # map faithfully to an array of one item type.
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            raise TypeError(
                f"Fixed heterogeneous tuple {tp!r} has no faithful single-`items` "
                "JSON Schema. Use a TypedDict (named fields) or list[...] for the "
                "output model, or add prefixItems support to _schema_for_type."
            )
        item_type = args[0] if args else Any
        return {"type": "array", "items": _schema_for_type(item_type)}
    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": _schema_for_type(value_type),
        }

    raise TypeError(
        f"Unsupported type annotation for schema generation: {tp!r}. "
        "Extend _schema_for_type in tools/_base.py if this construct is "
        "now used in the repo."
    )


@cache
def schema_from_typeddict(td: type) -> dict[str, Any]:
    """Generate a JSON Schema (``{"type": "object", ...}``) from a TypedDict.

    Every field is emitted under ``properties``. ``required`` reflects the
    two DISTINCT ways a field can be optional, both of which exist in the
    wire format: a ``NotRequired``/``total=False`` field may be ABSENT
    (omit-when-empty convention; e.g. ``GainAtPoint.phase_deg_unwrapped``),
    while an ``X | None`` field is always present but may be null. Marking
    an omitted key as required makes schema-validating MCP clients reject
    responses that follow the documented omit-when-empty behavior.
    """
    if not _is_typeddict(td):
        raise TypeError(f"Expected TypedDict, got {td!r}")

    hints = get_type_hints(td)
    # ``__required_keys__`` is computed at class-creation time and is UNRELIABLE
    # for ``NotRequired`` fields when the defining module uses ``from __future__
    # import annotations``: the wrapper is stringized, so the TypedDict metaclass
    # can't see it and wrongly counts the field as required (verified on 3.13).
    # ``get_type_hints(..., include_extras=True)`` EVALUATES the annotation,
    # recovering the ``NotRequired`` wrapper, so it detects optionality regardless
    # of stringization. (``Required`` in a ``total=False`` class is the symmetric
    # case but isn't used in this repo, so it's not special-cased.)
    extra_hints = get_type_hints(td, include_extras=True)
    structurally_required = getattr(td, "__required_keys__", frozenset(hints))
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name, field_type in hints.items():
        properties[field_name] = _schema_for_type(field_type)
        # A field is required unless the key may be absent entirely
        # (NotRequired / total=False) or its type admits None.
        admits_none = _is_union(field_type) and type(None) in get_args(field_type)
        not_required = get_origin(extra_hints.get(field_name)) is typing.NotRequired
        if field_name in structurally_required and not admits_none and not not_required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry for tool definitions and handlers."""

    def __init__(self) -> None:
        self._registered: list[RegisteredTool] = []

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_model: type[ToolInput] | None,
        annotations: types.ToolAnnotations,
        profiles: tuple[str, ...] = ("full",),
        output_schema: dict[str, Any] | None = None,
        output_model: type | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Register a tool and derive its schema from the input model.

        ``output_model`` (a TypedDict) is preferred over ``output_schema``
        (a hand-written dict): the schema is generated once at registration
        time from the same type the lib already returns, so the two can't
        drift. Only one of the two should be supplied.

        ``meta`` populates the tool definition's ``_meta`` object (e.g.
        ``{"ui": {"resourceUri": ...}}`` to declare an MCP Apps UI resource).
        """
        if output_model is not None and output_schema is not None:
            raise ValueError(
                f"Tool {name!r}: supply either output_model or output_schema, not both"
            )

        def decorator(handler: Callable) -> Callable:
            if any(rt.definition.name == name for rt in self._registered):
                raise ValueError(f"Tool already registered: {name}")

            @wraps(handler)
            async def wrapped(arguments: Any, state: SessionState) -> types.CallToolResult:
                if input_model is not None and not isinstance(arguments, input_model):
                    arguments = input_model.model_validate(arguments or {})
                return await handler(arguments, state)

            definition_kwargs: dict[str, Any] = {
                "name": name,
                "description": description,
                "inputSchema": (
                    _build_input_schema(input_model)
                    if input_model is not None
                    else {"type": "object", "properties": {}, "additionalProperties": False}
                ),
                "annotations": annotations,
            }
            if output_model is not None:
                definition_kwargs["outputSchema"] = schema_from_typeddict(output_model)
            elif output_schema is not None:
                definition_kwargs["outputSchema"] = output_schema

            definition = types.Tool(**definition_kwargs)
            if meta is not None:
                # ``meta`` is the field name; serializes by alias to ``_meta``.
                definition.meta = meta

            self._registered.append(
                RegisteredTool(
                    definition=definition,
                    handler=wrapped,
                    input_model=input_model,
                    profiles=frozenset(profiles),
                )
            )
            return wrapped

        return decorator

    def known_names(self) -> set[str]:
        """All registered tool names, across every profile (for diagnostics).

        Lets the dispatcher tell a profile-filtered tool (exists, hidden by the
        active profile) apart from a genuinely unknown name.
        """
        return {rt.definition.name for rt in self._registered}

    def get_for_profile(self, profile: str) -> tuple[list[types.Tool], dict[str, RegisteredTool]]:
        """Return the tool list and dispatch map for a profile."""
        effective_profile = profile if profile in {"full", "agentic"} else "full"
        tool_defs: list[types.Tool] = []
        tool_dispatch: dict[str, RegisteredTool] = {}
        for registered in self._registered:
            if effective_profile in registered.profiles:
                tool_defs.append(registered.definition)
                tool_dispatch[registered.definition.name] = registered
        return tool_defs, tool_dispatch


registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------


DEFAULT_PAGE_CAP = 50
"""Server-side ceiling on list-endpoint page size — the "caps at 50" the
``limit`` field descriptions document."""


def paginate(
    items: list, arguments: Any, cap: int = DEFAULT_PAGE_CAP
) -> tuple[list, int, int, int]:
    """Slice a list according to offset/limit from tool arguments.

    Returns:
        (page, total, offset, limit) tuple
    """
    total = len(items)
    offset = max(0, min(int(getattr(arguments, "offset", 0)), total))
    limit = min(int(getattr(arguments, "limit", cap)), cap)
    return items[offset : offset + limit], total, offset, limit


def pagination_metadata(total: int, offset: int, limit: int) -> dict[str, Any]:
    """Build structured pagination metadata for JSON responses."""
    has_more = offset + limit < total
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": offset + limit if has_more else None,
    }


# ---------------------------------------------------------------------------
# Simulation helpers — shared pre-checks
# ---------------------------------------------------------------------------


def require_simulator(state: SessionState) -> None:
    """Raise SimulationError if no simulator is available."""
    if state.default_simulator is None:
        raise SimulationError(no_simulator_message())


def resolve_netlist_path(netlist_str: str, state: SessionState) -> Path:
    """Resolve and validate a netlist path. Raises SimulationError on failure."""
    try:
        netlist_path = safe_path(netlist_str, state)
    except Exception as e:
        raise SimulationError(f"Invalid netlist path: {e}") from e
    if not netlist_path.exists():
        raise SimulationError(f"Netlist file not found: {netlist_path}")
    return netlist_path


def path_lock(registry: dict[Path, asyncio.Lock], path: Path, cap: int = 64) -> asyncio.Lock:
    """Get or create a per-path lock in ``registry``, LRU-bounded at ``cap``.

    Shared mechanism behind every per-file lock registry (schematic edits,
    ``.asc`` exports): refresh recency on hit; at capacity evict the oldest
    *unheld* lock — if all are held, overshoot temporarily rather than break
    mutual exclusion by evicting a lock someone is inside.
    """
    if path in registry:
        registry[path] = registry.pop(path)
        return registry[path]
    if len(registry) >= cap:
        for candidate in list(registry):
            if not registry[candidate].locked():
                del registry[candidate]
                break
    registry[path] = asyncio.Lock()
    return registry[path]


def circuit_lock_target(path: Path) -> Path:
    """Anchor for the cross-process lock on one circuit file.

    Lives under the circuit's ``.ltspice-mcp/locks/`` sidecar directory
    (``file_lock`` appends ``.lock``) so user directories aren't littered
    with lock files next to their circuits.
    """
    return path.parent / SIDECAR_DIRNAME / "locks" / path.name


@contextlib.asynccontextmanager
async def circuit_file_lock(path: Path) -> AsyncIterator[None]:
    """Cross-process lock for mutations/exports of one circuit file.

    Parallel MCP server processes editing the same circuit serialize here —
    without it, the whole-file read-modify-write saves are last-writer-wins
    and a concurrent session's edit is silently lost. Acquisition polls in a
    worker thread (per filelock's contract, so a contended lock never stalls
    the event loop); release is two fast syscalls, done inline.

    Acquire this BEFORE fetching a cached editor: the editor cache re-stats
    the file on every fetch, so taking the lock first guarantees the stat
    sees a concurrent writer's completed save rather than a mid-edit state.
    (Residual: on coarse-mtime filesystems like WSL's /mnt/c a same-size
    rewrite within one mtime tick can still go undetected — see FileCache.)
    """
    stack = contextlib.ExitStack()
    try:
        await asyncio.to_thread(stack.enter_context, file_lock(circuit_lock_target(path)))
    except TimeoutError as e:
        raise NetlistError(
            f"{path.name} is locked by another ltspice-mcp process "
            f"(waited {DEFAULT_TIMEOUT:.0f}s). Retry once its edit finishes."
        ) from e
    try:
        yield
    finally:
        stack.close()


# LTspice's ``create_netlist`` always writes the sidecar ``<name>.net`` next to
# the ``.asc``, so two concurrent exports of the same schematic would race on
# one output file (torn/partial reads of the deck). Serialize per resolved
# ``.asc`` path; distinct schematics still export in parallel.
_asc_export_locks: dict[Path, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def asc_export_lock(asc_path: Path) -> AsyncIterator[None]:
    """Serialize LTspice netlist exports of one schematic.

    In-process: a per-``.asc`` asyncio lock. Cross-process: the shared
    circuit file locks on BOTH the schematic and the sidecar ``.net`` —
    LTspice reads the ``.asc`` and overwrites the ``.net``, and a parallel
    session may be editing the ``.net`` itself under its own file lock.
    Fixed acquisition order (``.asc`` then ``.net``); edit paths take exactly
    one file lock, so no cycle is possible.
    """
    async with (
        path_lock(_asc_export_locks, asc_path),
        circuit_file_lock(asc_path),
        circuit_file_lock(asc_path.with_suffix(".net")),
    ):
        yield


async def resolve_runnable_netlist(netlist_str: str, state: SessionState) -> Path:
    """Resolve a path AND auto-export ``.asc`` → ``.net`` if needed.

    spicelib's ``SpiceEditor`` (used by the sweep / Monte Carlo runners)
    rejects ``.asc`` schematics — it expects the ``^*`` netlist comment
    header and otherwise fails with a cryptic ``Expected pattern "^\\*"
    not found``. This helper detects ``.asc`` and runs the LTspice
    ``create_netlist`` exporter to produce a sidecar ``.net``, so
    callers (sweep / MC config) can store the runnable path up front.

    The cheap safe_path/exists checks run inline, but the export launches the
    LTspice binary and blocks until it exits — heavy work that would stall the
    shared event loop, so it is offloaded via ``asyncio.to_thread``. It touches
    no cached editors, so the offload is safe under the concurrency contract.
    """
    netlist_path = resolve_netlist_path(netlist_str, state)
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path

    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        # Don't recommend export_netlist here — it ALSO needs LTspice, so that
        # advice dead-ends when only ngspice/etc. is available.
        raise SimulationError(
            f"{netlist_path.name} is an .asc schematic, which only LTspice can "
            "convert to a netlist, and LTspice is not available "
            f"(simulators: {list(state.available_simulators.keys())}). Supply a "
            "hand-written .cir/.net to simulate with the current simulator, or "
            "point the server at an LTspice executable ([simulator] path in the "
            "config file or LTSPICE_MCP_SIMULATOR_EXE) and restart. (The .asc's "
            "embedded .model/.lib/analysis directives can be reused in a .cir.)",
            show_hint=False,
        )
    async with asc_export_lock(netlist_path):
        try:
            net_path = Path(await asyncio.to_thread(ltspice_cls.create_netlist, str(netlist_path)))
        except Exception as e:
            raise SimulationError(
                f"Auto-exporting {netlist_path.name} to a netlist failed: {e}"
            ) from e
        if not await asyncio.to_thread(net_path.exists):
            raise SimulationError(f"Auto-export of {netlist_path.name} produced no .net file")
    return net_path


def inject_logopinfo(netlist_path: Path, simulator: type, job_id: str) -> Path:
    """Return a runnable netlist with ``.options logopinfo`` added, for LTspice ``.op`` runs.

    LTspice writes each semiconductor's small-signal operating point (gm, gds,
    vth, vdsat, junction caps) to the ``.log`` only under ``.options logopinfo``,
    and only for ``.op`` analyses — so adding it lets ``operating_point`` read
    those params back by name. ngspice uses ``@dev[param]`` raw traces instead
    and needs nothing here.

    Append-only into a per-job sibling file (a leading-dot, ``job_id``-stamped
    name) so the simulator sees the user's deck byte-for-byte plus the one
    directive; the original is never touched and relative ``.include``/``.lib``
    paths still resolve from the same directory. The ``job_id`` stamp keeps two
    concurrent or queued runs of the same netlist from clobbering each other's
    augmented copy; ``start_simulation`` deletes it once spicelib has staged the
    run. Returns the original path unchanged when injection doesn't apply
    (non-LTspice, non-text netlist, no ``.op``, or ``logopinfo`` already
    present) or the sibling can't be written.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    if not (isinstance(simulator, type) and issubclass(simulator, LTspice)):
        return netlist_path
    if netlist_path.suffix.lower() not in (".cir", ".net", ".sp"):
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    # Detect on the raw bytes (the directives are ASCII) — same plane the .end
    # splice below works on, so no decode round-trip is needed.
    if b"logopinfo" in data.lower():
        return netlist_path
    # ``.op\b`` excludes ``.options`` (the 't' blocks the word boundary); only a
    # real .op analysis emits the operating-point block. ``.dc`` does not.
    if not re.search(rb"(?im)^[ \t]*\.op\b", data):
        return netlist_path

    # Byte-level insertion before the final ``.end`` keeps the original encoding
    # intact (the added line is pure ASCII). ``.end\b`` skips ``.ends``.
    line = b".options logopinfo\n"
    ends = list(re.finditer(rb"(?im)^[ \t]*\.end\b.*$", data))
    if ends:
        at = ends[-1].start()
        augmented = data[:at] + line + data[at:]
    else:
        augmented = data + (b"" if not data or data.endswith(b"\n") else b"\n") + line

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{LOGOPINFO_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def safe_path(user_path: str, state: SessionState) -> Path:
    """Resolve and validate a user-provided path within security sandbox.

    This is a convenience wrapper around resolve_safe_path that uses
    the allowed_paths from the session state configuration.

    Args:
        user_path: Path string from user (relative or absolute)
        state: Current session state containing security configuration

    Returns:
        Resolved absolute path within sandbox

    Raises:
        PathSecurityError: If path violates security constraints
    """
    return resolve_safe_path(user_path, state.config.allowed_paths)


# ---------------------------------------------------------------------------
# Concurrency contract
#
# The MCP SDK dispatches EVERY incoming request as its own asyncio task on
# one shared event loop (mcp.server.lowlevel.Server.run start_soons a task
# per message), so tool handlers run concurrently. Anything that blocks the
# loop stalls every in-flight request — including cancel_job — and the
# transport's receive loop itself. Where work runs:
#
#   * Event loop (handler coroutines): argument validation, JobRegistry /
#     SessionState mutation, and ALL response building. format_response /
#     json_response / CallToolResult construction must stay in the handler
#     frame — the test suite's output-schema conformance hook attributes
#     emissions by walking the current thread's stack, so an emission from a
#     worker thread would silently skip schema validation. Offload
#     boundaries return plain data.
#   * asyncio.to_thread: heavy or potentially-slow filesystem work that is
#     read-only or atomic. The categories, one example each: result parsing
#     (services.load_raw), batch result/log loops (compute_batch_stats),
#     cross-process index writes (the recent-circuits touch: filelock poll
#     plus a durable fsync write), WSL interop (the first-call cmd.exe spawn
#     inside resolve_output_folder), and resource reads (the whole resource
#     router behind server.read_resource). Offloaded functions must stay
#     effect-free or atomic under cancellation: the awaiting task sees
#     CancelledError, but a worker thread that has started runs to
#     completion (cancellation cannot interrupt it mid-write); work
#     cancelled before the executor picks it up never begins. Either way
#     shared state stays consistent.
#   * Runner threads: long-lived simulator processes are owned by the
#     runner layer (sim_runner, sweep_runner, montecarlo_runner).
#
# Submit ordering: no suspension point between job registration/persistence
# (state.add_job / add_batch_job) and the asyncio.create_task that advances
# the job — a request cancelled at such an await would orphan a persisted
# "running"/"queued" job with no task behind it. Acquire runners (which
# await output-folder resolution) BEFORE registering the job.
#
# Intentionally inline on the loop, with bounds: netlist/schematic editor
# parses, mutations, and their cache invalidation (cached editor instances
# are MUTABLE and entangled with per-session snapshots — concurrent edits
# would be last-writer-wins data loss; worst case is a cold .asc parse over
# /mnt/c, ~1 s), job sidecar JSON loads (small per-circuit files), config
# saves (durable=False), log-file reads (KB scale), and library .lib parses
# (LibraryManager sessions are loop-owned mutable state; worst case ~1 s for
# a multi-MB vendor library).
# ---------------------------------------------------------------------------


# .include / .inc / .lib / .libfile <path> [extra]
_INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*\.(?:include|inc|lib|libfile)\b\s+(.+)$", re.IGNORECASE)


def _first_path_token(rest: str) -> str:
    """First (possibly quoted) path token of an include/lib directive's args."""
    rest = rest.strip()
    if rest[:1] in ("'", '"'):
        end = rest.find(rest[0], 1)
        if end != -1:
            return rest[1:end]
    parts = rest.split()
    return parts[0] if parts else ""


def _netlist_has_local_dependency(netlist_path: Path) -> bool:
    """True if the netlist pulls in a sibling file via a *relative* .include/.lib.

    Such a netlist can't be relocated to the run sidecar: a simulator resolves a
    relative include against the (now-moved) netlist's own directory, so the
    dependency would no longer be found. Bare library NAMES resolved via the
    simulator's own lib path (no matching local file) and absolute paths both
    survive relocation and don't count.
    """
    from ltspice_mcp.lib.encoding import read_spice_text

    try:
        text = read_spice_text(netlist_path)
    except OSError:
        return True  # unreadable — be conservative, keep it in place
    base = netlist_path.parent
    for line in text.splitlines():
        m = _INCLUDE_DIRECTIVE_RE.match(line)
        if not m:
            continue
        tok = _first_path_token(m.group(1))
        if not tok:
            continue
        # Absolute (POSIX, Windows drive, or UNC) paths survive relocation.
        if Path(tok).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", tok) or tok.startswith("\\\\"):
            continue
        if (base / tok).exists():
            return True
    return False


async def resolve_output_folder(state: SessionState, netlist_path: Path | None = None) -> Path:
    """Determine the output folder for the simulation runner.

    Kept **stable** — one ``{working_dir}/.ltspice-mcp/runs`` sidecar — so the
    single cached runner, ``cancel_job``, and the global ``max_parallel`` cap stay
    valid across runs. A per-deck output dir would change the folder on every run
    in a different directory, and ``RunnerManager`` invalidates the whole runner
    cache when the folder changes (losing in-flight process handles and splitting
    the concurrency semaphore per directory). Each run's artifacts are uniquely
    named (``{job_id}.*``), so they stay isolated within this shared folder; a
    caller finds them through the result path ``check_job`` reports.

    Two overrides:

    - **Relative ``.include``/``.lib`` deck:** the deck's own dir — the simulator
      resolves the relative path against the staged netlist's directory, so it
      can't be relocated (applies to single runs and sweeps/MC alike).
    - **WSL + LTspice + Linux-fs source:** a Windows-native temp dir. LTspice (a
      Windows process reaching the Linux fs over a ``wsl.localhost`` UNC share)
      can't write the SQLite ``.db`` behind ``.MEAS`` over UNC.

    Adds the chosen dir to allowed_paths so analysis tools can read results via
    safe_path(). The Windows temp-dir resolution spawns a cmd.exe interop
    subprocess on first call (memoized), so it runs via ``asyncio.to_thread`` — a
    wedged interop must not freeze the loop; the allowed_paths mutation stays on
    the loop after the await.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    from ltspice_mcp.lib.wsl import get_windows_output_dir, is_windows_native_path, is_wsl

    source_dir = netlist_path.parent if netlist_path is not None else state.working_dir
    has_local_dep = netlist_path is not None and _netlist_has_local_dependency(netlist_path)

    # Override: WSL + LTspice + Linux-fs source → Windows temp (UNC .db failure).
    if is_wsl() and not is_windows_native_path(source_dir) and not has_local_dep:
        sim_cls = state.default_simulator
        if sim_cls is not None and issubclass(sim_cls, LTspice):
            out = await asyncio.to_thread(get_windows_output_dir)
            if out is not None:
                if out not in state.config.allowed_paths:
                    logger.info(
                        f"WSL: routing LTspice output to {out} (source dir "
                        f"{source_dir} is on the Linux filesystem; .db/.MEAS "
                        "cannot write over UNC)"
                    )
                    state.config.allowed_paths.append(out)
                return out

    # Override: relative-include deck runs in its own dir so the include resolves.
    if has_local_dep:
        return source_dir

    # Default: one stable sidecar; per-job {job_id} naming isolates each run.
    runs = state.working_dir / ".ltspice-mcp" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if runs not in state.config.allowed_paths:
        state.config.allowed_paths.append(runs)
    return runs
