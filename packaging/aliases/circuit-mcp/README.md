# circuit-mcp

This package is an alias for [**ltspice-mcp**](https://pypi.org/project/ltspice-mcp/),
a SPICE circuit-simulation MCP server (LTspice and ngspice) for Claude and
other LLMs.

Installing this package installs and runs `ltspice-mcp`:

```bash
uvx circuit-mcp          # run the server
pip install circuit-mcp  # installs ltspice-mcp as a dependency
```

`ltspice-mcp` is the canonical package. Documentation, configuration, issues,
and source live there: https://github.com/cognitohazard/ltspice-mcp

## Migration from 0.5

Version 0.6.0 replaced the `full` and `agentic` tool profiles with one set of
8 tools. A `[tools] profile` setting left in an old config is ignored. To keep
the old 49-tool set, pin the 0.5 series: `ltspice-mcp==0.5.*`.
