"""MCP protocol logging and progress notifications via ContextVar.

The ContextVars are set in server.py:call_tool() before handler dispatch,
making MCP-level notifications available to tools and services without
passing server/session references through every function signature.

Usage in tools/services:
    from ltspice_mcp.lib.mcp_logging import mcp_log, mcp_progress

    await mcp_log("info", "Simulation started")
    await mcp_progress(3, total=10, message="Running sweep step 3/10")
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, Literal

LogLevel = Literal["debug", "info", "notice", "warning", "error", "critical"]

# These are set by server.py:call_tool() before dispatching to handlers.
# Typed as Any to avoid importing MCP session types into domain code.
_mcp_log_fn: ContextVar[Any] = ContextVar("mcp_log_fn", default=None)
_mcp_progress_fn: ContextVar[Any] = ContextVar("mcp_progress_fn", default=None)


async def mcp_log(level: LogLevel, message: str) -> None:
    """Send an MCP log message to the client, if available.

    Safe to call from any async context — silently no-ops if called
    outside a tool dispatch (e.g., during startup or from a background thread).
    """
    fn = _mcp_log_fn.get(None)
    if fn is not None:
        await fn(level, message)


async def mcp_progress(
    completed: float, total: float | None = None, message: str | None = None
) -> None:
    """Send an MCP progress notification to the client, if available.

    Only emits if the client provided a progressToken in the request.
    Safe to call even when no token was provided — silently no-ops.
    """
    fn = _mcp_progress_fn.get(None)
    if fn is not None:
        await fn(completed, total, message)


def set_log_fn(fn: Callable[..., Coroutine] | None) -> None:
    """Set the MCP log function for the current context (called by server.py)."""
    _mcp_log_fn.set(fn)


def set_progress_fn(fn: Callable[..., Coroutine] | None) -> None:
    """Set the MCP progress function for the current context (called by server.py)."""
    _mcp_progress_fn.set(fn)
