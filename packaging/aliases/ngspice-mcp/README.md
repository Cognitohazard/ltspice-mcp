# ngspice-mcp

Alias for [**ltspice-mcp**](https://pypi.org/project/ltspice-mcp/) — a SPICE
circuit-simulation MCP server (LTspice and ngspice) for Claude and other LLMs.
The server runs ngspice as a first-class engine; LTspice is also supported.

Installing this package installs and runs `ltspice-mcp`:

```bash
uvx ngspice-mcp          # run the server
pip install ngspice-mcp  # installs ltspice-mcp as a dependency
```

`ltspice-mcp` is the canonical package. Documentation, configuration, issues,
and source live there: https://github.com/cognitohazard/ltspice-mcp

## Migration from 0.5

Version 0.6.0 removed the `full` and `agentic` tool profiles; the server now
serves one consolidated surface (7 tools). A `[tools] profile` config naming a
removed profile warns and serves that surface. To keep the old 49-tool
surface, pin the 0.5 series: `ltspice-mcp==0.5.*`.
