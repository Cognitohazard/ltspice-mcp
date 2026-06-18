"""Entry point for ltspice-mcp server.

Runs the MCP server with stdio transport (standard MCP pattern).
The server communicates over stdin/stdout, with logs going to stderr.
"""

import argparse
import asyncio
import os
from io import TextIOWrapper

import anyio
from mcp.server.stdio import stdio_server

from ltspice_mcp.server import register_init_options, server


def main():
    """Entry point for ltspice-mcp server.

    This is the main entry point called by `uv run ltspice-mcp` or
    `python -m ltspice_mcp`. It sets up the stdio transport and runs
    the MCP server event loop.

    The server uses:
    - stdin: MCP protocol messages (JSON-RPC)
    - stdout: MCP protocol responses (JSON-RPC)
    - stderr: Logging output (startup summary, errors, diagnostics)

    Before starting, fd 0 (stdin) is redirected to /dev/null at the OS
    level so that any subprocess spawned by spicelib or WSL helpers
    inherits /dev/null instead of the MCP JSON-RPC pipe. The real stdin
    fd is dup'd to a new descriptor and passed to the MCP transport.

    This works around python-sdk#671: stdio_server() doesn't protect
    stdin/stdout from subprocess inheritance.
    """
    parser = argparse.ArgumentParser(description="LTSpice MCP Server")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to ltspice-mcp.toml config file (default: CWD or $LTSPICE_MCP_CONFIG)",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["LTSPICE_MCP_CONFIG"] = args.config

    # Redirect fd 0 BEFORE any imports that might spawn subprocesses.
    # 1. Dup the real stdin fd to a new descriptor
    real_stdin_fd = os.dup(0)
    # 2. Replace fd 0 with /dev/null at the OS level
    devnull_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull_fd, 0)
    os.close(devnull_fd)

    # Now subprocesses inheriting fd 0 get /dev/null.
    # The MCP transport will read from real_stdin_fd.
    asyncio.run(_run(real_stdin_fd))


async def _run(real_stdin_fd: int):
    """Run the MCP server with stdio transport.

    Args:
        real_stdin_fd: File descriptor for the real stdin pipe
            (already dup'd away from fd 0, which is now /dev/null).
    """
    # Wrap the saved fd into a file object for the MCP transport
    real_stdin_file = os.fdopen(real_stdin_fd, "rb")
    real_stdin = anyio.wrap_file(TextIOWrapper(real_stdin_file, encoding="utf-8"))

    async with stdio_server(stdin=real_stdin) as (read_stream, write_stream):
        # Share the init options with the lifespan so it can rewrite the
        # instructions to name the detected simulators (see server.py).
        init_options = server.create_initialization_options()
        register_init_options(init_options)
        await server.run(read_stream, write_stream, init_options)
