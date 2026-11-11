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

- **Structured**: tool outputs declare schemas, and results are typed data
  instead of free text.
- **Validated**: tools that change a file refuse invalid states before
  writing it.
- **Geometry-aware**: `.asc` schematic operations work with the positions
  of pins, parts and wires on the sheet as well as with component names and
  nets.
- **LTspice-specific**: the server handles `.asc`/`.asy`/`.raw` behavior and
  Windows interop directly. ngspice is fully supported for simulation, result
  parsing, diagnostics and analysis; it has no schematic format, so the
  geometry tools do not apply to it. QSPICE and Xyce are supported on a
  best-effort basis through spicelib's common interface.

### Platforms

Most users run the server natively on Windows alongside LTspice. WSL2 and
Linux are also supported. The server uses WSL interop (the `/mnt` drive
mapping, `wslpath`, the Windows-side temp directory for `.MEAS`) only when it
detects WSL.

|simulator|platform|
|-|-|
|LTspice|Windows native; Linux through Wine; WSL through Windows interop|
|ngspice|Linux and Windows native|
|QSPICE|Windows native only (runs are refused under WSL or Wine)|
|Xyce|Linux and Windows native|

macOS is not supported yet. The server reads every `.raw` and `.log` file in
a parser process with a hard memory limit and cleans up that process and any
it started (`lib/parser_process.py`). Neither has been validated on macOS, so
the server refuses every result read there with `unsupported_platform`.
[docs/TESTING.md](TESTING.md) lists the tests that supporting it needs.

A schematic (`.asc`) runs through the netlist LTspice exports from it, which
is in LTspice's own dialect. The server prepares that netlist for ngspice
only, by removing the `.backanno` line and the `§` LTspice puts in some
instance names, so a schematic run on QSPICE or Xyce is refused and needs a
hand-written netlist.
ngspice runs in spicelib's default compatibility mode, `kiltpsa`, which drops
the section name from a `.lib <file> <section>` line that selects a corner;
set `[simulator] ngbehavior` (for example `hsa`) when a deck uses one.

### Why an MCP server instead of spicelib scripts

The server is for an AI agent working on circuits on the machine that has the
simulator and the circuit files. Its audiences, in order:

- **Coding agents**, such as Claude Code, Cursor and Codex. They have file
  tools and a shell, and work on that machine directly, through Claude Code's
  Remote Control, or in a cloud container that runs ngspice. This is the main
  audience.
- **People chatting in Claude Desktop.** They see charts in the chat. The
  assistant there has no file tools unless the person installs one, so it
  writes a netlist through `run_code`.

Scripts and notebooks use the same code through the Python API.

An agent that can run code could write spicelib scripts itself. The server
helps in these cases:

- **The simulator and the files where they live.** The server finds and runs
  LTspice or ngspice on that machine (natively, under Wine or across WSL),
  reads their result files and writes the files LTspice opens, so the agent
  handles no detection, interop or file formats itself.
- **Reliability.** A tool call runs a tested code path. LLM-written spicelib
  code has import errors and wrong method names, and it forgets to save the
  netlist.
- **Context.** A tool call is about 50 tokens; the same work in Python is 20
  to 40 lines.
- **Structured analysis by default.** The `analyze_results` recipes return
  typed numbers for -3 dB points (`bode_filter`), for slopes, gains and
  crossings (`bode_slope`, `bode_point`, `bode_crossing`) and for transient
  shape (`signal_stats`, `edges`, `transient_response`), so common questions
  need no image.
- **No user setup.** Install the server once. The project's environment
  needs no spicelib, and the user needs no Python.

Tools and Python split the work. A tool does what the server must own: a job
that outlives a call, a transactional schematic edit, a verification, a single
read whose reply is used as it is, and a chart for a person. Python does
loops, combining runs and math on traces: `run_code` runs a snippet against
the same code, and a script uses it directly as `Api(working_dir=...)`, which
reads and writes the same job records as the server.

## Design principles

### Validate before write

Tools that change a file check the change first and refuse a state they can
prove invalid. A refusal names the segments, pins or labels that blocked it,
so the agent can choose again instead of guessing. Agents recover poorly from
mistakes, and a refusal saves the turns that recovery would take.

The same holds for every refusal, finding and error, whether or not a write
was involved: it names the field, the place and the way round, such as the
fields an argument accepts, the coordinates of a collision, or the run that
gives a result this one cannot. Tests pin what a refusal names
(`tests/test_validation_errors.py` for argument refusals), generate a
misspelled, missing and misplaced argument for every tool
(`tests/test_refusals_name_the_fix.py`), and refuse any reply in the suite
whose finding gives no line, coordinates or part reference (`tests/conftest.py`).

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
`verify_circuit` has LTspice export a saved sheet to a netlist and compares
that with a reference netlist or schematic, to report any unintended change.

Even when every op succeeds, the reply includes sheet-level warnings such as
floating pins, duplicate wire segments and dangling net labels, so the caller
does not need a separate call to inspect the sheet.

A check must say what it did not look at: a comparison that leaves part of a
circuit out says which part, a list cut short says how to read the rest, and
a check that did not run is reported as not run. A verdict that reads as
complete while it covered only part of the circuit is worse than no verdict.
`tests/test_check_scope_is_stated.py` and
`tests/test_capped_lists_say_how_to_continue.py` hold the tools to this.

### Structured outputs

Tools declare an output schema and return `structuredContent` beside their
text, and the test suite checks every reply it produces against its tool's
schema. The tool listing leaves the output schemas out, because they would be
its largest part and each reply shows its own structure.

A reply leaves out every field that says nothing: an empty list or object, a
null, a false flag, and a sub-answer its overall answer already gives. So a
reply about work that found nothing costs no tokens for saying so. A missing
field means "checked, and nothing to report", never "not checked": a check
that did not run or a value that could not be read is always stated. One
function applies the rule to every reply, and each tool lists the few fields
it keeps anyway, with the reason ([mcp_surface.md](design/mcp_surface.md),
"Fields left out").

### A short tool listing

A client loads every tool definition before its first call, and those tokens
stay in context for every later turn. Each argument description is therefore
short (the unit, the sign convention, the default, the interaction with
sibling fields), and the details are in a section of the packaged guide or in
[mcp_surface.md](design/mcp_surface.md). `[tools] listing` sets how much is
sent. The default, `compact`, omits argument descriptions, which a model can
read with `inspect(kind="reference")`, and `full` sends them (mcp_surface.md,
"Two listing modes").

### Results report facts

Results report measured conditions and the simulator's own messages. They
give no verdict on whether a result can be trusted and no name for a
waveform's shape. A result reports a fact such as a crest factor (peak over
RMS) of 8.1 for one stretch of a waveform, and leaves conclusions such as
"spike" or "unreliable" to the model.
A result has a severity only when the simulator assigned one.

### Scalars and waveform shape

By default the analysis tools return scalars, such as time-weighted RMS,
rise time or phase margin, because a correct scalar is easier for a model to
reason over than an array. When the waveform shape matters, three levels of
detail are available:

- **A recipe for a known shape**, such as `edges` for rise and fall times
  or `transient_response` for a step response, returns a number for it.
  Prefer one when it exists.
- **The `waveform` recipe** returns the min, max, rms and mean of each bucket
  of samples, so decimation keeps a narrow spike. To zoom in, call it again
  with a narrower `window`. With `format: "csv"` it writes every sample to
  disk and returns the path.
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

The HTML file is for terminal clients such as Claude Code, which do not show
images to the user and pass a PNG only to the model. Each unit gets its own
chart panel, so volts and amps never share a y-axis. The chart plots every
sample of a trace up to 100,000 points (the `max_points` argument); above that
it keeps the minimum and maximum of each interval and reports that it
downsampled. The PNG is off by default (`[analysis] attach_plot`), costs
about 1,000 tokens for two panels, and needs the optional `raster` extra. The
core install adds no plotting dependency; the chart library is bundled.

### One call per intent

The caller states what it wants, and the server does the steps in between:
staging a copy, taking locks, exporting, comparing, running. One
`edit_schematic` batch edits a sheet and checks it against a reference;
repeating a `request_id` returns the earlier job, so comparing two settings
takes one call; `jobs(action="place_beside_sheet")` puts a run's results where
LTspice opens them. When a task takes a sequence of calls, the fix is an op or
an option, not a documented sequence. `tests/test_one_call_per_task.py` counts
the calls each of these takes.

### A short reply, the data behind an id

A reply carries a summary that is true on its own. The bulk stays on disk,
recorded per job and case and addressed by id and path, for a later call or
the Python API to read and reduce, so an old run reloads without simulating
again. A reply should name the call that reads more, and a result too large
to read whole should be readable in part
(`tests/test_short_reply_data_behind_an_id.py`).

### Repeatable requests

The same `request_id` returns the same job ("Backend: spicelib" below), so a
retry is safe. A client keeps its own copy of the tool definitions, so a
renamed argument breaks it until it reloads them. Renames are therefore made
together, in one release, and the refusal of an old name says so
(`tests/test_argument_names_are_kept.py`).

### LTspice's own tools as peers

The server writes the files LTspice reads (a sheet's netlist, its `.plt` plot
settings, a run's results beside its sheet) and shows a sheet or a run in
LTspice's own window, rather than imitating them ("The file on disk is
authoritative" below). `tests/test_files_ltspice_reads_are_recorded.py` ties
each such file to a recording of LTspice reading it.

### Evidence before assumption

The server models only LTspice behaviour recorded from both supported builds
([TESTING.md](TESTING.md), "Recorded LTspice behaviour"). Unrecorded
behaviour is where the server goes wrong, so a recording comes before the
code that relies on it. `tests/test_ltspice_readers_are_recorded.py` requires
every module that reads an LTspice format to name its recording.

### Fast enough to try

Small runs stay fast enough that an agent runs an experiment instead of
reasoning one out, and work added to the run path is weighed against that.
Tests do not time anything (`CLAUDE.md`, "Working practice");
`scripts/benchmark_small_runs.py` compares small runs with a local baseline.

### The file on disk is authoritative

An LTspice window shows a copy of the file. LTspice reads a sheet once and
keeps that copy, so saving from the window overwrites any edit made on disk.
On Windows with LTspice 26.1 or later, the server connects to open windows
through the bridge that LTspice installs. `edit_schematic` refuses to commit
while a window's copy differs from the file, and updates the window after a
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

The server has 8 registered tools. Six of them form three pairs by purpose:

- **Execute**: `run_experiments` and `jobs`.
- **Understand**: `analyze_results` and `inspect`.
- **Author**: `edit_schematic` and `verify_circuit`.

The other two are `plot_waveform`, the interactive waveform chart, and
`run_code`, which runs a Python snippet in a separate Python process that the
server starts on first use and keeps for the session, with the Python API already
loaded (`[tools] run_code = false` removes it). Releases before 0.6.0 offered
other tool sets with one tool per operation; a deployment that needs those
pins `ltspice-mcp==0.5.*`. `inspect(kind="capabilities")` reports
`tool_profile: "consolidated"`, the name of the current and only tool set.

Each of the six covers many operations, which the caller describes as data in
its arguments instead of calling a separate tool for each. Sweeps, corners
and Monte Carlo are `run_experiments` `variations`; each analysis is an
`analyze_results` recipe; each schematic change is an `edit_schematic` op;
each read is an `inspect` kind. No tool edits netlist text, since an agent
with file access does that directly.

Schematic editing is one tool with many ops (`add_component`,
`move_component`, `move`, `remove_component`, `set_component_value`,
`set_component_attribute`, `wire_pins`, `add_net_label`, `remove_net_label`,
`remove_wire`, `add_directive`, `remove_directive`, `replace_directive` and
`set_plot_panes`).
`move` shifts parts, wires, labels and text by one offset, and
`set_plot_panes` writes the waveform panes LTspice opens for the sheet. Every
tool's schema uses model context whether or not the tool is called, so one
tool that takes every kind of op is cheaper than one tool per op, and it lets
a caller batch ops into one transaction. The ops work on pin coordinates,
bounding boxes and the topology of named nets. `wire_pins` refuses diagonal
wires, pin collisions, shorts between named nets and a waypoint that would
join another net's wiring, and accepts a point on an existing wire as an
endpoint for a T-junction.
`add_component` returns the placed pins and bounding box.
`inspect(kind="symbol")` returns the same geometry without placing anything,
for planning an edit.

### Backend: spicelib

The server builds on [spicelib](https://github.com/nunobrum/spicelib)
(>= 1.4.9, < 1.6), the Python library for SPICE automation by Nuno Brum. It
supplies the simulator classes and batch runner, the `.asc` schematic editor,
and the readers for `.raw` waveforms and `.MEAS` log output. The server
depends on spicelib directly rather than on PyLTSpice, which re-exports it.
The server implements Monte Carlo perturbation itself, and perturbed cases
run through the same execution path as every other run.

Each job records the simulator executable it launches (path, size,
modification time and SHA-256), and each case (one simulation in the job)
records the build version its run reported in its output, so results from
two builds stay distinguishable. A submission that repeats a request under
the same `request_id` returns the existing job, unless the build it would
launch now differs from the recorded one; then it is refused as an
`idempotency_conflict`.

## WSL support

On WSL, the server runs the Windows LTspice through Windows interop instead
of Wine. Set `simulator.path` in `ltspice-mcp.toml` to the Windows-side
executable, for example `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`;
spicelib cannot detect it across the WSL boundary. The server handles the
rest:

- Paths are converted with `wslpath` instead of Wine's `Z:` prefix.
- Symbol-library paths under `%LOCALAPPDATA%` are resolved through
  `cmd.exe`, since they vary by Windows user profile.
- Simulation output is written to a Windows temp directory when the working
  directory is on the Linux file system, because LTspice fails to write its
  `.MEAS` database on `\\wsl.localhost\...` paths, and the measurement
  results would be lost.
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

A reply with no `observations` means that no check fired; it leaves out an
empty list, as it leaves out every field that says nothing. It does not mean
that the result is correct.

LTspice and ngspice get full diagnostics, with ngspice's console output read
alongside its log. For QSPICE and Xyce, the server parses convergence and
fatal errors on a best-effort basis.

## Non-goals

The server leaves these out by design, and none of them is planned.

- **Layout or routing the caller did not ask for.** The caller sets each
  part's position and each wire's path; a refusal returns the conflict set.
- **Semantic part search** ("find me a low-Vgs-th NMOS under 100mΩ
  Rds(on)"). It needs every `.model` card parsed, units normalized across
  vendors and an index of parameters that can be queried, which is a separate
  project.
- **Thin wrappers around file edits.** No tool edits netlist text, because a
  client with native file editing does it better, and a client without it can
  still pass the server a deck it wrote.
- **Per-simulator parity for geometry.** LTspice and ngspice are equal for
  simulation, result parsing, diagnostics and analysis. The geometry layer
  (`.asc`, `.asy`, symbol coordinates) is LTspice-only, since ngspice has no
  schematic format to edit.
- **Auth, multi-tenancy and remote MCP support.** This is a local tool, with
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
7. Include a custom library with a `.lib` or `.include` directive and use one
   of its models in a new circuit.

If a step fails, check `simulator.path` in the TOML and, when
`ltspice-mcp.toml` sets `[security] allowed_paths`, confirm that it includes
your working directory.

[docs/TESTING.md](TESTING.md) lists the platforms the test suite is released
on and the recorded LTspice behaviour the tests check the server against.
