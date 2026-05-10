"""Shared utilities for tool handlers."""

import copy
import json
import logging
import types as _stdlib_types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from mcp import types
from pydantic import BaseModel, ConfigDict

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib.pathutil import resolve_safe_path
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
    if origin is list or origin is tuple:
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

    Every field is emitted under ``properties``. ``required`` lists fields
    that don't accept None — optional-by-convention is expressed as
    ``X | None`` on the TypedDict, not via ``NotRequired``, so the repo
    has a single way to spell "may be missing" and the schema reflects it.
    """
    if not _is_typeddict(td):
        raise TypeError(f"Expected TypedDict, got {td!r}")

    hints = get_type_hints(td)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name, field_type in hints.items():
        properties[field_name] = _schema_for_type(field_type)
        # A field is required unless its type admits None.
        admits_none = _is_union(field_type) and type(None) in get_args(field_type)
        if not admits_none:
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
    ) -> Callable[[Callable], Callable]:
        """Register a tool and derive its schema from the input model.

        ``output_model`` (a TypedDict) is preferred over ``output_schema``
        (a hand-written dict): the schema is generated once at registration
        time from the same type the lib already returns, so the two can't
        drift. Only one of the two should be supplied.
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

            self._registered.append(
                RegisteredTool(
                    definition=types.Tool(**definition_kwargs),
                    handler=wrapped,
                    input_model=input_model,
                    profiles=frozenset(profiles),
                )
            )
            return wrapped

        return decorator

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


def paginate(items: list, arguments: Any, cap: int = 50) -> tuple[list, int, int, int]:
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
        raise SimulationError(
            "No simulator available. Check server status.\n\n"
            f"Available simulators: {list(state.available_simulators.keys())}"
        )


def resolve_netlist_path(netlist_str: str, state: SessionState) -> Path:
    """Resolve and validate a netlist path. Raises SimulationError on failure."""
    try:
        netlist_path = safe_path(netlist_str, state)
    except Exception as e:
        raise SimulationError(f"Invalid netlist path: {e}") from e
    if not netlist_path.exists():
        raise SimulationError(f"Netlist file not found: {netlist_path}")
    return netlist_path


def resolve_runnable_netlist(netlist_str: str, state: SessionState) -> Path:
    """Resolve a path AND auto-export ``.asc`` → ``.net`` if needed.

    spicelib's ``SpiceEditor`` (used by the sweep / Monte Carlo runners)
    rejects ``.asc`` schematics — it expects the ``^*`` netlist comment
    header and otherwise fails with a cryptic ``Expected pattern "^\\*"
    not found``. This helper detects ``.asc`` and runs the LTspice
    ``create_netlist`` exporter to produce a sidecar ``.net``, so
    callers (sweep / MC config) can store the runnable path up front.
    """
    netlist_path = resolve_netlist_path(netlist_str, state)
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path

    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        raise SimulationError(
            f"{netlist_path.name} is an .asc schematic and the active runner "
            "needs a netlist. Run ltspice_export_netlist first, or configure "
            "LTspice as a simulator. Available: "
            f"{list(state.available_simulators.keys())}"
        )
    try:
        net_path = Path(ltspice_cls.create_netlist(str(netlist_path)))
    except Exception as e:
        raise SimulationError(
            f"Auto-exporting {netlist_path.name} to a netlist failed: {e}"
        ) from e
    if not net_path.exists():
        raise SimulationError(f"Auto-export of {netlist_path.name} produced no .net file")
    return net_path


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
# MCP stdio transport processes one request at a time — tool handlers run
# synchronously on the event loop.  All spicelib parser/editor calls are
# fast enough to run inline.  Long-lived simulation work uses
# asyncio.to_thread() in the runner layer (sim_runner, sweep_runner,
# montecarlo_runner), not here.
# ---------------------------------------------------------------------------


def resolve_output_folder(state: SessionState) -> Path:
    """Determine the output folder for simulation files.

    On WSL, if the working dir is on the Linux filesystem (not /mnt/),
    uses a Windows-native temp dir instead. This ensures LTspice can write
    its .db (SQLite) files, which is required for .MEAS results to appear
    in .log files.

    Also adds the output dir to allowed_paths so analysis tools can read
    the result files via safe_path().
    """
    from ltspice_mcp.lib.wsl import get_windows_output_dir, is_windows_native_path, is_wsl

    if is_wsl() and not is_windows_native_path(state.working_dir):
        win_dir = get_windows_output_dir()
        if win_dir is not None:
            if win_dir not in state.config.allowed_paths:
                logger.info(
                    f"WSL: using Windows output dir {win_dir} "
                    f"(working_dir {state.working_dir} is on Linux filesystem)"
                )
                state.config.allowed_paths.append(win_dir)
            return win_dir

    return state.working_dir
