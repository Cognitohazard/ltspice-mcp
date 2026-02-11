"""Entry point for ltspice-mcp server.

Runs the MCP server with stdio transport (standard MCP pattern).
The server communicates over stdin/stdout, with logs going to stderr.
"""

import asyncio

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

    Exits when stdin is closed or the server encounters a fatal error.
    """
    asyncio.run(_run())


async def _run():
    """Run the MCP server with stdio transport.

    Creates read/write streams from stdin/stdout and runs the server
    event loop. The server's lifespan context manager handles startup
    and shutdown.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
