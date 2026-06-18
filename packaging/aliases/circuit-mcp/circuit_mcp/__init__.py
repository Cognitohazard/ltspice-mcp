"""Alias package for ltspice-mcp.

Installing ``circuit-mcp`` pulls in the canonical ``ltspice-mcp`` distribution
and runs the same server, self-identifying as ``circuit-mcp`` in the MCP
handshake. See https://github.com/cognitohazard/ltspice-mcp.
"""

import os

__all__ = ["main"]


def main() -> None:
    # Identify as this alias in serverInfo.name. Set before importing the server
    # (its name is bound at import time); an explicit override still wins.
    os.environ.setdefault("LTSPICE_MCP_SERVER_NAME", "circuit-mcp")
    from ltspice_mcp.main import main as _main

    _main()
