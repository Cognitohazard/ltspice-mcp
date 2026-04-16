"""MCP protocol logging notifications via ContextVar.

The ContextVar is set in server.py:call_tool() before handler dispatch,
making MCP-level notifications available to tools and services without
passing server/session references through every function signature.

Usage in tools/services:
    from ltspice_mcp.lib.mcp_logging import mcp_log

    await mcp_log("info", "Simulation started")
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, Literal

LogLevel = Literal["debug", "info", "notice", "warning", "error", "critical"]

_mcp_log_fn: ContextVar[Any] = ContextVar("mcp_log_fn", default=None)


async def mcp_log(level: LogLevel, message: str) -> None:
    """Send an MCP log message to the client, if available.

    Safe to call from any async context — silently no-ops if called
    outside a tool dispatch (e.g., during startup or from a background thread).
    """
    fn = _mcp_log_fn.get(None)
    if fn is not None:
        await fn(level, message)


def set_log_fn(fn: Callable[..., Coroutine] | None) -> None:
    """Set the MCP log function for the current context (called by server.py)."""
    _mcp_log_fn.set(fn)
