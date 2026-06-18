"""Entry point for the LTspice-MCP Desktop Extension (type="uv" MCPB bundle).

The host runs ``uv run --directory <bundle> server/run.py`` (see ``mcp_config``
in manifest.json); uv reads the bundle's ``pyproject.toml``, installs
``ltspice-mcp`` and its native dependencies (numpy/scipy) into an on-demand
environment, then runs this file. Dependencies are resolved per host rather
than vendored: numpy/scipy ship as per-platform binary wheels, so vendoring
would lock the bundle to one OS and Python ABI; uv fetches the right wheels for
the host instead.

Running this file directly also starts the server when ``ltspice-mcp`` is
importable in the active environment.
"""

from ltspice_mcp.main import main

if __name__ == "__main__":
    main()
