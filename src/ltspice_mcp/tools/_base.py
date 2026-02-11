"""Shared utilities for tool handlers."""

import asyncio
from pathlib import Path
from typing import Any, Callable, TypeVar

from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.state import SessionState

T = TypeVar("T")


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
