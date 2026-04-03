"""Shared utilities for tool handlers."""

import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

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
    text: str, data: dict[str, Any], fmt: str | None = None
) -> types.CallToolResult:
    """Return a CallToolResult with text content and structuredContent.

    Always populates structuredContent for programmatic access.
    The format param controls the text representation:
    - "json": text is JSON-formatted (for clients that parse text)
    - "text" or None: text is human-readable (default)
    """
    if fmt == "json":
        return json_response(data)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=data,
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
    ) -> Callable[[Callable], Callable]:
        """Register a tool and derive its schema from the input model."""

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
            if output_schema is not None:
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
