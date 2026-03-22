"""Shared utilities for tool handlers."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from mcp import types
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import ResultError, SimulationError
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

FORMAT_PROP: dict[str, Any] = {
    "type": "string",
    "enum": ["json", "text"],
    "description": "Response format: 'json' for structured data, 'text' for human-readable (default: text)",
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

RO_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def paginate(
    items: list, arguments: dict, cap: int = 50
) -> tuple[list, int, int, int]:
    """Slice a list according to offset/limit from tool arguments.

    Returns:
        (page, total, offset, limit) tuple
    """
    total = len(items)
    offset = int(arguments.get("offset", 0))
    limit = min(int(arguments.get("limit", cap)), cap)
    return items[offset: offset + limit], total, offset, limit


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
# Analysis helpers — shared raw-file loading and validation
# ---------------------------------------------------------------------------

async def load_raw(raw_path: Path, state: SessionState) -> RawRead:
    """Load and cache a RawRead instance. Raises ResultError on failure."""
    try:
        return await run_sync(
            state.results.get, raw_path,
            lambda p: RawRead(str(p), traces_to_read="*"),
        )
    except FileNotFoundError:
        raise ResultError(f"Result file not found: {raw_path}")
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )


async def validate_signal(raw: RawRead, signal: str) -> None:
    """Validate that a signal exists in the raw file. Raises ResultError."""
    from ltspice_mcp.lib.raw_parser import get_trace_names
    trace_names = await run_sync(get_trace_names, raw)
    if signal not in trace_names:
        available = ", ".join(trace_names[:10])
        if len(trace_names) > 10:
            available += f", ... ({len(trace_names)} total)"
        raise ResultError(
            f"Signal '{signal}' not found. Available signals: {available}"
        )


async def validate_step(raw: RawRead, step: int) -> None:
    """Validate that a step index is in range. Raises ResultError."""
    from ltspice_mcp.lib.raw_parser import get_step_count
    step_count = await run_sync(get_step_count, raw)
    if step < 0 or step >= step_count:
        raise ResultError(
            f"Step {step} out of range. Valid range: 0 to {step_count - 1}"
        )


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
        raise SimulationError(f"Invalid netlist path: {e}")
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


async def run_sync(fn: Callable[..., T], *args: Any) -> T:
    """Run a synchronous blocking function in a thread pool.

    All blocking spicelib calls MUST go through this wrapper to avoid
    blocking the asyncio event loop. This is critical for server responsiveness
    when multiple operations are happening concurrently.

    Args:
        fn: Synchronous function to call
        *args: Arguments to pass to the function

    Returns:
        Result from the function call
    """
    return await asyncio.to_thread(fn, *args)


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
            logger.info(
                f"WSL: using Windows output dir {win_dir} "
                f"(working_dir {state.working_dir} is on Linux filesystem)"
            )
            resolved = win_dir.resolve()
            if not any(p.resolve() == resolved for p in state.config.allowed_paths):
                state.config.allowed_paths.append(win_dir)
            return win_dir

    return state.working_dir
