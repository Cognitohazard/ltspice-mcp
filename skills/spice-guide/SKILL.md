---
name: spice-guide
description: >
  Use for any circuit simulation or SPICE task: LTspice, ngspice, netlists,
  .asc schematics, analog or power circuits, simulation results. Read this
  before writing a deck, running a simulator, or writing analysis code
  yourself: it points to the ltspice-mcp server's guide.
---

# Read the server's guide first

The ltspice-mcp server carries the guide for this work: how to use the
server, when to use Python or the tools, the Python API, the rules that cause
silent errors, and sections on each simulator, the tools, and whole tasks such
as characterizing an amplifier. Read its core once per session, before you
start:

- With the MCP tools: `inspect(queries=[{"kind": "guide"}])`
- In Python, in `run_code` or a script with `from ltspice_mcp.api import Api`:
  `print(Api.guide())`
- From a shell where the package is installed, without starting the engine:
  `python -m ltspice_mcp.api guide`

The core ends in an index. Before each task, read the section the index names
for it, by passing its name: `inspect(queries=[{"kind": "guide", "section":
"ltspice"}])`, or `Api.guide("ltspice")`.
