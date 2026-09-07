"""Entry point for ltspice-mcp server.

Runs the MCP server with stdio transport (standard MCP pattern).
The server communicates over stdin/stdout, with logs going to stderr.
"""

import argparse
import asyncio
import os

from mcp.server.runner import serve_dual_era_loop
from mcp.server.stdio import stdio_server

from ltspice_mcp.server import server


def main():
    """Entry point for ltspice-mcp server.

    This is the main entry point called by `uv run ltspice-mcp` or
    `python -m ltspice_mcp`. It sets up the stdio transport and runs
    the MCP server event loop.

    The server uses:
    - stdin: MCP protocol messages (JSON-RPC)
    - stdout: MCP protocol responses (JSON-RPC)
    - stderr: Logging output (startup summary, errors, diagnostics)

    ``stdio_server()`` serves the protocol from private duplicates of fd 0 and
    fd 1 and points the descriptors themselves at the null device and stderr,
    so a subprocess spawned by spicelib or a WSL helper cannot read from or
    write to the JSON-RPC pipe.
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

    asyncio.run(_run())


async def _run():
    """Run the MCP server with stdio transport.

    This is ``Server.run()`` with the initialization options left unset, which
    is what makes the handshake read live state: the runner then builds them
    per ``initialize`` request, after the lifespan has filled in the detected
    simulators, instead of from a snapshot taken before the server booted.
    ``server/discover`` reads the same live state on its own.
    """
    async with (
        stdio_server() as (read_stream, write_stream),
        server.lifespan(server) as lifespan_context,
    ):
        await serve_dual_era_loop(
            server,
            read_stream,
            write_stream,
            lifespan_state=lifespan_context,
        )
