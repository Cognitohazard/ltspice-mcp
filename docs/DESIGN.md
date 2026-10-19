# ltspice-mcp — Design

This document covers the server's scope, architecture and design principles.

Install and client setup are in [README.md](../README.md) and
[USAGE.md](USAGE.md). Tool arguments and replies are specified in
[docs/design/mcp_surface.md](design/mcp_surface.md), the Python API in
[docs/design/python_api.md](design/python_api.md), and the netlist parser in
[docs/spice_lex.md](spice_lex.md). Current bugs and limitations are in the
GitHub issue tracker.

## Scope

The server targets **structured, validated, geometry-aware,
LTspice-specific** operations:

- **Structured**: tool outputs declare schemas; results are typed, not free
  text.
- **Validated**: tools that change a file refuse invalid states before
  writing it.
- **Geometry-aware**: `.asc` schematic operations work in symbol
  coordinates, not just netlist tokens.
- **LTspice-specific**: the server handles `.asc`/`.asy`/`.raw` behavior and
  Windows interop directly. ngspice is fully supported for simulation, result
  parsing, diagnostics and analysis; it has no schematic format, so the
  geometry tools do not apply to it. QSPICE and Xyce are best-effort through
  spicelib's common interface.

### Platforms

Most users run the server natively on Windows alongside LTspice. WSL2 and
Linux are also supported. The server uses WSL interop (the `/mnt` drive
mapping, `wslpath`, the Windows-side temp directory for `.MEAS`) only when it
detects WSL.

|simulator|platform|
|-|-|
|LTspice|Windows native; Linux and macOS through Wine; WSL through Windows interop|
|ngspice|Linux, macOS and Windows native|
|QSPICE|Windows native only (runs are refused under WSL or Wine)|
|Xyce|Linux, macOS and Windows native|

A schematic (`.asc`) runs through LTspice's netlist export, which is adapted
only for ngspice, so QSPICE and Xyce runs need a hand-written netlist.
ngspice runs in spicelib's default compatibility mode, `kiltpsa`, which drops
the section from a sectioned `.lib <file> <section>` corner select; set
`[simulator] ngbehavior` (for example `hsa`) when a deck uses one.

### Why an MCP server, and not spicelib scripts

An LLM with code execution could write spicelib scripts itself. The server
helps in these cases:

- **Clients without code execution.** Claude Desktop, ChatGPT and web chat
  clients have no shell or Python, so MCP tools are their only access to
  simulation. This is the primary audience.
- **Reliability.** A tool call runs a tested code path. LLM-written spicelib
  code has import errors, wrong method names and missing netlist saves.
- **Context.** A tool call is about 50 tokens; the same work in Python is 20
  to 40 lines.
- **Structured analysis by default.** The `analyze_results` recipes return
  typed numbers: `bode_filter` for -3 dB points, `bode_slope`, `bode_point`
  and `bode_crossing` for slopes, gains and crossings, and `signal_stats`,
  `edges` and `transient_response` for transient shape. Common questions need
  no image.
- **No user setup.** Install the server once. The project's environment
  needs no spicelib, and the user needs no Python.

A fixed tool set is less flexible than arbitrary Python. For loops and
complete results, `run_code` runs a Python snippet against the same engine,
and a script can use the engine directly as `Api(working_dir=...)`, which
reads and writes the same job records as the server.

## Design principles

### Validate before write

Tools that change a file check the change first and refuse a state they can
prove invalid. A refusal names the segments, pins or labels that blocked it,
so the agent can choose again instead of guessing. Agents recover poorly from
mistakes, and a refusal saves the turns that recovery would take.

### No auto-routing

The server does not lay out or route a circuit; the caller decides the
drawing. When the server refuses a wire, it returns the conflict set (each
blocking element and its position), and the caller picks another route.

### Safe edits

`edit_schematic` applies a batch of ops as one transaction, written by atomic
rename, so a failed batch leaves the sheet unchanged. `expected_sha256` names
the revision the batch was prepared against; if another writer committed
first, the call returns `revision_conflict` and writes nothing. `dry_run`
resolves and validates a batch and returns its geometry without writing.
`verify_circuit` exports a committed sheet on a copy and compares it with a
reference netlist or schematic, which reports any unintended change.

Even when every op succeeds, the reply includes sheet-level warnings such as
floating pins, duplicate wire segments and dangling net labels, which saves a
follow-up inspection call.

### Structured outputs

Tools declare an output schema and return `structuredContent` beside their
text. A client that runs code against tool calls can use the schemas as
types. The tool listing omits the schemas: they were its largest block, and
each reply carries its own structure.

### A short tool listing

A client loads every tool definition before its first call, and those tokens
stay in context for every later turn. Each argument description is therefore
short (the unit, the sign convention, the default, the interaction with
sibling fields), and the details are in a section of the packaged guide or in
[mcp_surface.md](design/mcp_surface.md). `[tools] listing` sets how much is
sent: `compact`, the default, omits argument descriptions, and `full` sends
them (mcp_surface.md, "Two listing modes").

### Report facts, not verdicts

Results report measured conditions and the simulator's own messages, with no
trust verdict and no shape label: "crest factor 8.1, 5.2 times the median
bucket" is a fact, while "spike" or "unreliable" is a conclusion for the
model to draw. A result carries a severity only when the simulator assigned
one.

### Waveforms: scalars first, shape when it matters

By default the analysis tools return scalars, such as time-weighted RMS,
rise time or phase margin, because a correct scalar is easier for a model to
reason over than an array. When the waveform shape matters, three layers
serve it:

- **Detectors** return a number for a known shape. Prefer one when it
  exists.
- **The `waveform` recipe** returns a decimated stat-envelope: min, max, rms
  and mean per bucket, so decimation keeps a narrow spike. To zoom in, call
  it again with a narrower `window`. With `format: "csv"` it writes every
  sample to disk and returns the path.
- **`plot_waveform`** draws an interactive chart and returns its file path
  and a summary of each trace (min and max and where they occur, first and
  final values, and on a transient the time-weighted mean), computed from
  every sample before any decimation, so the model gets the chart's numbers
  without viewing it.

Each consumer has its own channel:

|consumer|channel|
|-|-|
|the agent, for computation|the `waveform` recipe in `format: "csv"`, written to disk|
|a vision model|a PNG attached to a `plot_waveform` result (`attach_plot`)|
|a person|`plot_waveform`: an in-chat widget on an MCP Apps host such as Claude Desktop, otherwise a self-contained HTML file opened on the desktop|
|a person in LTspice|`plot_waveform(in_ltspice=true)`, which opens the run's results in the LTspice window|

The HTML file is for terminal clients, which do not show images to the user:
Claude Code passes a PNG to the model only. Each unit gets its own chart
panel, so volts and amps never share a y-axis. The chart plots every sample
unless the count exceeds a high cap, in which case it downsamples and reports
that. The PNG is off by default (`[analysis] attach_plot`), costs about 1,000
tokens for two panels, and needs the optional `raster` extra. The core
install adds no plotting dependency; the chart library is bundled.

### The file on disk is authoritative

An LTspice window shows a copy of the file. LTspice reads a sheet once and
keeps that copy, so saving from the window overwrites any edit made on disk.
On Windows with LTspice 26.1 or later, the server reaches open windows
through the bridge LTspice installs: `edit_schematic` refuses to commit while
a window's copy differs from the file, and updates the window after a
commit. `inspect(kind="open_in_ltspice")` lists the open documents.
`verify_circuit(in_ltspice=true)` and `plot_waveform(in_ltspice=true)` show a
sheet or a run in LTspice, and start LTspice for that only when no LTspice
window is open (`[schematic] start_ltspice`). Every other LTspice process the
server starts on Windows runs on a hidden desktop and does not take keyboard
focus (`[simulator] hidden_desktop = false` turns this off). See USAGE.md,
"Working with LTspice open".

## Architecture at a glance

```
MCP protocol layer    server.py, resources.py: lifespan, request routing, MCP resources
Tool layer            tools/*.py: tool definitions and handlers
Core logic            lib/*.py: runs and jobs, result parsing, analysis, schematic editing
Config / state        config.py, state.py, errors.py
```

### The tool surface

The server has 8 registered tools. Six are operations over three planes:

- **Execute**: `run_experiments` and `jobs`.
- **Understand**: `analyze_results` and `inspect`.
- **Author**: `edit_schematic` and `verify_circuit`.

The other two are `plot_waveform`, the interactive waveform chart, and
`run_code`, which runs a Python snippet in a warm worker that holds the
engine (`[tools] run_code = false` removes it).
`inspect(kind="capabilities")` reports `tool_profile: "consolidated"`, the
only profile.

Each of the six takes a declarative payload rather than a fixed argument
list. Sweeps, corners and Monte Carlo are `run_experiments` `variations`;
each analysis is an `analyze_results` recipe; each schematic change is an
`edit_schematic` op; each read is an `inspect` kind. No tool edits netlist
text, since an agent with file access does that directly. Releases before
0.6.0 had one tool per operation; a deployment that needs those pins
`ltspice-mcp==0.5.*`.

Schematic editing is one tool with many ops (`add_component`,
`move_component`, `remove_component`, `set_component_value`,
`set_component_attribute`, `wire_pins`, `add_net_label`, `remove_net_label`,
`remove_wire`, `add_directive`, `remove_directive`, and `set_plot_panes`,
which writes the waveform panes LTspice opens for the sheet). Every tool's
schema uses model context whether or not the tool is called, so one tool
with an op union is cheaper than one tool per op, and it lets a caller batch
ops into one transaction. The ops work on pin coordinates, bounding boxes and
named-net topology. `wire_pins` refuses diagonal wires, pin collisions,
named-net shorts and a waypoint that would join another net's wiring, and
accepts a point on an existing wire as an endpoint for a T-junction.
`add_component` returns the placed pins and bounding box.
`inspect(kind="symbol")` returns the same geometry without placing anything,
for planning an edit.

### Backend: spicelib

The server builds on [spicelib](https://github.com/nunobrum/spicelib)
(>= 1.4.9, < 1.6), the Python library for SPICE automation by Nuno Brum. It
supplies the simulator classes and batch runner, the `.asc` schematic editor,
and the readers for `.raw` waveforms and `.MEAS` log output. The server
depends on spicelib directly, not on PyLTSpice, which re-exports it. The
server implements Monte Carlo perturbation itself, and perturbed cases run
through the same execution path as every other run.

Each job records the simulator executable it launches (path, size,
modification time and SHA-256), and each case records the build version its
run reported in its output, so results from two builds stay distinguishable.
A reused `request_id` that would now launch a different build is refused as
an `idempotency_conflict`.

## WSL support

On WSL, the server runs the Windows LTspice through Windows interop, not
Wine. Set `simulator.path` in `ltspice-mcp.toml` to the Windows-side
executable, for example `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`;
spicelib cannot detect it across the WSL boundary. The server handles the
rest:

- Paths are converted with `wslpath` instead of Wine's `Z:` prefix.
- Symbol-library paths under `%LOCALAPPDATA%` are resolved through
  `cmd.exe`, since they vary by Windows user profile.
- Simulation output goes to a Windows temp directory when the working
  directory is on the Linux file system, because LTspice's `.MEAS` database
  writes fail on `\\wsl.localhost\...` paths and measurement results would be
  lost.
- A netlist keeps its extension (`.cir`, `.net`, `.sp`), which LTspice
  requires.

## Diagnostics

The server reads each simulator log and attaches its warnings and errors
(parse errors with caret pointers, fatal errors, convergence messages,
`.MEAS` parse failures) to the replies of `run_experiments`, `jobs` and the
`summary` recipe as `warnings` and `errors` lists, so the agent reads a
failure in the tool reply instead of searching a log.

The run summary also has an `observations` list:

|observation kind|contents|
|-|-|
|`relay`|the simulator's own error lines, with the simulator's severity; the server never assigns one|
|`reconciliation`|requested `.meas`/`.four` outputs that were not produced|
|`value`|non-finite samples and extreme node values in the trace data|
|`coverage`|skipped checks, so an incomplete result is not mistaken for a clean one|

An empty `observations` list means no check fired; it does not mean the
result is verified correct.

LTspice and ngspice get full diagnostics, with ngspice's console output read
alongside its log. QSPICE and Xyce get best-effort parsing of convergence and
fatal errors.

## Non-goals

These are deliberate gaps, not pending features.

- **Layout or routing the caller did not ask for.** The caller sets each
  part's position and each wire's path; a refusal returns the conflict set.
- **Semantic part search** ("find me a low-Vgs-th NMOS under 100mΩ
  Rds(on)"). It needs every `.model` card parsed, units normalized across
  vendors and a queryable parameter index, which is a separate project.
- **Thin wrappers around file edits.** No tool edits netlist text: a client
  with native file editing does it better, and a client without it can still
  pass the server a deck it wrote.
- **Per-simulator parity for geometry.** LTspice and ngspice are equal for
  simulation, result parsing, diagnostics and analysis. The geometry layer
  (`.asc`, `.asy`, symbol coordinates) is LTspice-only, since ngspice has no
  schematic format to edit.
- **Auth, multi-tenancy and remote-MCP plumbing.** This is a local tool, with
  no OAuth flows and no per-tenant isolation.
- **Generic cross-simulator EDA features** that PyLTSpice and ngspice's CLI
  already cover.

## Configuration

Settings are in `ltspice-mcp.toml` in the working directory. The server
writes it with defaults on the first tool call if it is missing
(`LTSPICE_MCP_WRITE_CONFIG=false` turns that off). An environment variable
with the `LTSPICE_MCP_` prefix overrides a TOML value. The sections are
`[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[logging]`,
`[schematic]`, `[tools]` and `[state]`.

The server keeps its records, runs, result sets, export snapshots and plots
in `.ltspice-mcp/` in the working directory, and the recent-circuits index
and lock files in a per-user folder (`%LOCALAPPDATA%\ltspice-mcp` on
Windows). USAGE.md, "Configuration" and "Files the server creates", lists
the options and every file the server writes.

## Verification recipe

End-to-end smoke test for a fresh install:

1. Create an RC lowpass netlist (R = 1k, C = 100n, so fc is about 1.59 kHz).
2. Add a `.ac dec 100 1 1Meg` directive.
3. Run it with `run_experiments`.
4. Read it back with a `bode_filter` recipe and check that the -3 dB point
   is near 1.59 kHz.
5. Change R to 10k (fc about 159 Hz).
6. Run and measure again, and check that the bandwidth moved.
7. Pull in a custom library with a `.lib` or `.include` directive and use one
   of its models in a new circuit.

If a step fails, check `simulator.path` in the TOML and, when
`ltspice-mcp.toml` sets `[security] allowed_paths`, confirm that it includes
your working directory.

[docs/TESTING.md](TESTING.md) lists the platforms the test suite is released
on and the recorded LTspice behaviour the tests check the server against.
