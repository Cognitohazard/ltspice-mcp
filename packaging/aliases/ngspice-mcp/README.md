# ngspice-mcp

This package is an alias for [**ltspice-mcp**](https://pypi.org/project/ltspice-mcp/),
a SPICE circuit-simulation MCP server for Claude and other LLMs. It supports
both ngspice and LTspice.

Installing this package installs and runs `ltspice-mcp`:

```bash
uvx ngspice-mcp          # run the server
pip install ngspice-mcp  # installs ltspice-mcp as a dependency
```

`ltspice-mcp` is the canonical package. Documentation, configuration, issues,
and source live there: https://github.com/cognitohazard/ltspice-mcp

## Migration from 0.5

Version 0.6.0 removed the `full` and `agentic` tool profiles; the server now
exposes one set of 7 tools. If `[tools] profile` in the config names a removed
profile, the server logs a warning and exposes those same 7 tools. To keep the
old 49-tool set, pin the 0.5 series: `ltspice-mcp==0.5.*`.
