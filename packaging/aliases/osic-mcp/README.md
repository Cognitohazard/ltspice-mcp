# osic-mcp

This package is an alias for [**ltspice-mcp**](https://pypi.org/project/ltspice-mcp/),
an MCP server and Python library for circuit and open-source IC design. It
runs SPICE simulations with ngspice and LTspice, runs sweeps and Monte Carlo
as jobs, returns measurements as numbers, and edits LTspice schematics.

Installing this package installs and runs `ltspice-mcp`:

```bash
uvx osic-mcp          # run the server
pip install osic-mcp  # installs ltspice-mcp as a dependency
```

`ltspice-mcp` is the canonical package. Documentation, configuration, issues,
and source are at https://github.com/cognitohazard/ltspice-mcp
