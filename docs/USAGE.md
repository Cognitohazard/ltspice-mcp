# Using ltspice-mcp

This guide covers the setup options, LTspice window features, configuration
and Python API that the [README](../README.md) leaves out.

- [Installing](#installing)
- [Making sure the assistant uses the server](#making-sure-the-assistant-uses-the-server)
- [Working with LTspice open](#working-with-ltspice-open)
- [Alongside LTspice's own MCP server](#alongside-ltspices-own-mcp-server)
- [Configuration](#configuration)
- [Files the server creates](#files-the-server-creates)
- [WSL](#wsl)
- [PNG rendering](#png-rendering)
- [Where the server can run](#where-the-server-can-run)
- [An example session](#an-example-session)
- [Python API](#python-api)

## Installing

`ltspice-mcp --help` confirms the install worked. The same program is also
published as `circuit-mcp`, `ngspice-mcp` and `osic-mcp`, in case one of
those names is easier to remember.

The Claude Code plugin and the Claude Desktop extension fetch the server with
`uv` and include the `raster` extra for [PNG rendering](#png-rendering). If
you start the server with `uvx` yourself, use
`uvx --from 'ltspice-mcp[raster]' ltspice-mcp` to get it too.

## Making sure the assistant uses the server

Some clients show an assistant only a tool's name until it picks one, so an
assistant may run the simulator from the shell instead.

**The server's name.** The assistant sees every tool prefixed with it, as in
`mcp__spice__run_experiments`, so a name that includes SPICE helps. It is the
key in your client's MCP config, or the word after `claude mcp add`. The plugin
uses `spice`, and the Desktop extension uses its manifest name,
`ltspice-mcp`. If yours is something generic like `sim1`, rename it. Rename
`ltspice` too, because LTspice's own MCP server uses that name.

**An instruction.** Put the rule from the README in your project's
`CLAUDE.md`, or your client's equivalent file. The rule is absolute because
an assistant told to weigh the choice usually picks the shell it already
knows.

### When the shell is fine

If you drop the rule, an assistant with a shell can run quick, one-off
ngspice simulations directly. They are scriptable and usually take under a
second, so the server adds little for them. Use the server for LTspice runs,
values read by name from binary `.raw` files, sweeps and Monte Carlo, jobs
that outlive one call, and checked `.asc` editing. `analyze_results` can also
measure a `.raw` file produced outside the server, so a simulation run from
the shell can still be analyzed here.

## Working with LTspice open

These features need Windows and LTspice 26.1 or later. The server reaches
your LTspice window through `ltspice-mcp-bridge.exe`, a program LTspice
installs.

**Edits to an open schematic.** LTspice does not reload a schematic that
changes on disk. If the server changed only the file, the window would keep
showing the old version, and saving it would undo the assistant's edit. So
when the assistant edits a schematic you have open, the server also updates
the window. The change appears at once, and Ctrl+Z undoes it. If the window
has unsaved changes, the edit is refused and the assistant asks you to save
or close the schematic first. Editing never starts LTspice; it only updates a
window that is already open. `[schematic] sync_open_window = false` turns
this off.

**Showing you things in LTspice.** The assistant can see which schematic you
have in front, so you can refer to it as "this circuit". When you ask, it can
open a schematic in LTspice, or open a simulation run with the traces you
asked about already plotted. If LTspice isn't running, it starts it for these
two requests only; `[schematic] start_ltspice = false` turns that off.

When the run is of a schematic, clicking a net in the window plots it, as if
you had run the schematic in LTspice. To make that work, the server copies
the run's results and log next to the schematic as `<name>.raw` and
`<name>.log`, replacing any files of those names, and opens them from the
schematic. It doesn't simulate again.

**LTspice's documentation.** For questions about LTspice itself, such as a
shortcut or a menu, the assistant reads the reference files LTspice 26.1 and
later installs.

None of this needs LTspice's own MCP server.

## Alongside LTspice's own MCP server

LTspice 26.1 and later ships its own MCP server. On Windows it offers to add
itself to Claude Code, Claude Desktop, Copilot and Cursor as `ltspice`.

The two do different jobs, and you can install both. LTspice's server
controls the LTspice window, so it can read a schematic you haven't saved and
run it on screen while you watch. It returns raw samples and the log, so
working out a phase margin or setting up a Monte Carlo run is left to the
assistant, and it can only edit a schematic by replacing its whole text. This
server works on the files instead. It places and wires parts with the
geometry checked, runs sweeps and Monte Carlo as background jobs, returns
measurements as numbers, and also runs ngspice.

**A setting worth changing on LTspice's side.** When LTspice's server has no
window to use, it starts a hidden copy of LTspice, one per assistant session,
and changes made through it can end up in a copy you can't see. To prevent
that, register it with `--ltspice-path` set to a file that doesn't exist; it
can then only use a window you have open. This workaround is not a
documented option, so check it again after updating LTspice.

**If you registered this server as `ltspice`,** rename it to `spice`. The
Claude Code plugin has already been renamed, so its tool names changed once,
from `…_ltspice__run_experiments` to `…_spice__run_experiments`. Update any
saved permission rule or instruction that uses the old names.

## Configuration

No configuration is required. To change settings, copy
[`ltspice-mcp.example.toml`](../ltspice-mcp.example.toml) to
`ltspice-mcp.toml` in the working directory; it lists every option with its
default. `--config PATH` or `LTSPICE_MCP_CONFIG` picks another file, and any
setting can be overridden with an `LTSPICE_MCP_`-prefixed environment
variable. The settings you are most likely to change:

```toml
[simulator]
default = "ltspice"      # ltspice, ngspice, qspice, xyce (unset = auto-detect)
path = ""                # explicit executable path (required on WSL)
ngbehavior = "hsa"       # ngspice compatibility mode; "hsa" makes sectioned .lib corner selection work
hidden_desktop = true    # Windows: run LTspice where its window can't take your keyboard focus; false shows it

[simulator.executables]  # more builds, chosen per run as execution.simulator = "ltspice:xvii"
# xvii = "C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"

[security]
# allowed_paths = ["."]  # a sandbox: the only folders the tools may use; unset = no sandbox

[simulation]
# max_parallel = 4       # default: number of CPU cores, capped at 8
timeout = 300.0          # seconds, for LTspice netlist export
# run_timeout = 3600     # seconds per run when a request sets none; default: no limit

[schematic]
sync_open_window = true  # Windows, LTspice 26.1+: show edits in the LTspice window that has the schematic open
start_ltspice = true     # start LTspice when asked to show something in it and it isn't running

[tools]
listing = "compact"      # "full" sends every argument description, about 45% more to load
text_channel = "summary" # "json" repeats each reply's data as text, for clients that ignore structuredContent
run_code = true          # false removes run_code

[state]
persist_jobs = true
```

**`listing`.** The default, `compact`, leaves argument descriptions out of
the tool list, which makes the list a session loads before its first call
about 45% smaller. The tools accept the same calls either way. The assistant
looks up an argument with `inspect(kind="reference", query="...")`, and a
rejected call lists the fields it accepts. `full` sends every description.

**`text_channel`.** A tool reply carries its data as `structuredContent` and
a one-line summary as text. Claude Code and other clients that read
`structuredContent` give the assistant the data. A client that passes only
the text gives it the summary alone, so the assistant sees no results; set
`text_channel = "json"` for such a client, which puts the data in the text
too.

**`run_code`** runs a Python snippet in a worker process where `api` is a
[Python API](#python-api) `Api` open on the server's working directory and
configuration, for loops over runs and numpy on samples. The snippet runs
with the server's own file and process permissions, which `allowed_paths`
does not confine. Approve `mcp__spice__run_code` in your client as you would a
shell, and don't allow it as part of a blanket `mcp__spice__*` rule. Set
`run_code = false` if more than one client can reach the server, for example
through a proxy. The change takes effect at the next start;
`inspect(kind="capabilities")` reports whether the tool is on.

**Plots.** `plot_waveform` opens its chart in a browser when the client can't
show it in the chat; `[analysis] open_plot = false` returns only the file
path. `[analysis] attach_plot = true` also returns a PNG of the chart for the
assistant to look at (about 1,000 tokens per call), which needs
[PNG rendering](#png-rendering).

## Files the server creates

Beside your circuits, the server writes only files you ask for (an edited
schematic, an exported netlist, a plot saved to a folder you name), plus:

- the `<name>.net` that LTspice writes next to a schematic it runs;
- the `<name>.raw`, `<name>.log` and `<name>.plt` (the traces to plot)
  placed next to a schematic when you ask to see its run in LTspice. A
  `<name>.plt` saved from LTspice is left as it is.

Everything else goes in two places:

- **`.ltspice-mcp/` in the working directory:** job records, runs, results,
  the netlists exported from schematics, and plots. Add it to `.gitignore`.
- **A per-user folder:** the recent-circuits list, lock files that keep two
  sessions from editing one file at once, and the drawings `verify_circuit`
  renders, each deleted a week after it was made. It is
  `%LOCALAPPDATA%\ltspice-mcp` on Windows and `~/.local/state/ltspice-mcp`
  (or under `$XDG_STATE_HOME`) elsewhere; `LTSPICE_MCP_HOME` moves it.

Two environment variables reduce the files written to the working directory:

- `LTSPICE_MCP_WRITE_CONFIG=false` stops the first tool call from writing a
  default `ltspice-mcp.toml` there.
- `LTSPICE_MCP_STORE_DIR=<dir>` keeps `.ltspice-mcp/` in `<dir>` instead (one
  subfolder per working directory). A server, script or Python API session
  then finds those job records only if it has the same setting.

## WSL

On WSL, the server runs the Windows LTspice directly (not through Wine), but
it can't find it on its own. Set the path:

```toml
[simulator]
path = "/mnt/c/Program Files/ADI/LTspice/LTspice.exe"
```

Simulation output goes to a Windows temp folder automatically, because
LTspice loses `.MEAS` results when it writes to the Linux file system.

`.asc` editing reads symbols from the library of the LTspice build the
server runs, which it finds automatically on Windows, under Wine and on WSL.
Folders in `[schematic] symbol_paths` or `LTSPICE_MCP_SYMBOL_PATHS` are
searched before that library. A symbol saved in the same folder as the
schematic is always found first, as in LTspice.

## PNG rendering

`verify_circuit` draws schematics as SVG, but an assistant can only see a
drawing returned as a PNG, since clients don't reliably display SVG. PNG
needs the `raster` extra, which the plain install leaves out. Install the
server with it:

```bash
uv tool install 'ltspice-mcp[raster]'    # or: pipx install 'ltspice-mcp[raster]'
```

With `uvx`, use `uvx --from 'ltspice-mcp[raster]' ltspice-mcp`. The Claude
Code plugin and the Claude Desktop extension already include it. The extra's
rasterizer, `resvg-py`, ships as a prebuilt wheel and needs no other library.
Restart the server afterwards. The image `plot_waveform` attaches with
`attach_plot` needs the same extra.

Without the extra, the server uses `cairosvg` if it is installed, together
with the Cairo library it loads. If there is no rasterizer, a PNG request
returns an SVG file instead, and the reply names what is missing and how to
install it. A `verify_circuit` render with `delivery` set to `"inline"` or
`"both"` then returns the SVG markup in `render.svg`, and a plot still
returns its chart and trace summaries.
`inspect(kind="capabilities")` reports the same before anything is drawn:

```json
"render": {"png": false, "missing": "extra", "reason": "...", "remedy": "..."}
```

`missing` is `"extra"`, or `"native_library"` when only `cairosvg` is
installed and the Cairo library is not. `missing`, `reason` and `remedy` are
null when `png` is true.

## Where the server can run

The server runs the simulator and reads circuit files locally, so it must run
on the machine that has them. That can be your own machine with a local MCP
client (Claude Desktop, Claude Code, Cursor, Gemini CLI, Codex, …), or a
cloud agent whose sandbox can install ngspice and register the server (tested
with Claude). LTspice is a desktop app, so it only works locally; ngspice
works in either place.

A web chat with no sandbox can't run it. To use one, connect it to a machine
you control with a stdio-to-HTTP bridge such as
[`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy). Only expose the server
on a network you fully control, because it writes files and starts processes,
and only `allowed_paths` together with `run_code = false` confines it to
chosen folders. [SECURITY.md](../SECURITY.md) describes the threat model.

## An example session

For "design a 1 kHz RC low-pass and verify it", the assistant writes the
netlist (R = 1k, C = 159.155n, so fc = 1 kHz):

```spice
* rc.cir — RC low-pass
V1 in 0 AC 1
R1 in out 1k
C1 out 0 159.155n
.ac dec 50 1 1Meg
.end
```

then calls two tools:

```
verify_circuit(path="rc.cir", checks=["syntax"])
  → outcome "complete", no findings

run_experiments(
  circuits=[{"path": "rc.cir"}],
  analyze={"recipes": [{"key": "lp", "metric": "bode_filter", "signal": "V(out)"}]},
)
```

The `lp` measurement comes back as (abridged):

```json
{
  "signal": "V(out)",
  "filter_type": "lowpass",
  "passband_gain_db": 0.0,
  "passband_ripple_db": 0.02,
  "cutoff_low_hz": null,
  "cutoff_high_hz": 1000.4,
  "stopband_rejection_db": 59.97,
  "rolloff_slope_db_per_decade": -19.9,
  "estimated_order": 1,
  "warnings": []
}
```

If the result is off target, the assistant edits the netlist and runs it
again. A long simulation returns a job ID instead of waiting, and
`jobs(action="status" | "wait" | "cancel")` checks on the job, waits for it
or cancels it. Jobs, signals, measurements and the configuration are also
readable as MCP resources (`spice://results/...`, `spice://netlists/...`,
`spice://config`).

Sweeps and Monte Carlo are set in the `variations` argument of
`run_experiments` and have no tools of their own.

- `assign` sets a `.param`, a component's value (so a supply voltage is an
  assignment to its source), `REF@model` to swap one device's model, or
  `X1:delvto` for an offset on a transistor inside a subcircuit. Several
  entries run every combination, and so do the lists within one entry unless
  it sets `combine: "zip"`, which steps them together. Corners are run by
  swapping models.
- `random` adds Monte Carlo runs with component tolerances, `.param` and
  `.MODEL` parameter spread, and Pelgrom W·L device mismatch.

Variations do not set the temperature. Set it in the deck with `.temp`,
`.step temp` or `.options temp=`. A `.param TEMP` is rejected, because SPICE
never uses it as the simulation temperature.

Every measurement, `inspect` kind, schematic operation and check is listed
by `inspect(kind="reference")`, generated from the tools themselves. The
design of each tool is in
[design/mcp_surface.md](design/mcp_surface.md).

## Python API

The [README](../README.md#python-api) has a first example, and
[design/python_api.md](design/python_api.md) specifies the full contract.

| | MCP server | Python API |
|-|-|-|
| Called by | an assistant in an MCP client | a script, notebook or CI job, often one an assistant wrote |
| Results | large results come in pages, continued with a cursor | complete results, waveforms as numpy arrays |
| Long runs | the server keeps the job running; follow it with `jobs` | jobs stop when the script exits, unless it detaches them |
| Good for | interactive work such as exploring, editing and running a few checks | optimizers, custom post-processing, pipelines |

A sweep or Monte Carlo set is one call either way. Use the API when each run
depends on code that processes the previous result.

**Both at once.** The server and scripts work in the same directory on the
same job records, so either can read a job the other started by its
`job_id`.

**Jobs a script starts.** `api.close()`, the end of a `with` block, or normal
interpreter exit cancels the script's unfinished jobs.
`run_experiments(wait=False, detach=True)` instead hands the job to a
separate process that supervises it. The call returns once the job is
recorded, the receipt names that process and its log, and the job keeps
running after the script exits. Read or cancel it later by `job_id`, from a
script or the server.

**Differences from the tools:**

- Results are complete, because the API collects every page. It removes the
  MCP-only `budget` and `execution.wait_s` and lists each in the result's
  `warnings`. It rejects a page cursor or continuation field with
  `ApiValidationError`; `raw_page=True` accepts them and returns one page, as
  the tools do.
- A process can have only one open `Api`, so creating a second one before
  the first is closed, or one inside a running server process, raises
  `ApiSessionError`. A new `Api` starts in well under a second.
- `api.load_raw()` and `api.measurements()` return numpy data for your own
  processing.

**Reference and guide.** `api.reference()` lists the six operations, and
`api.reference("run_experiments")` returns the arguments of one operation.
`api.guide()` returns the guide; pass a section name to get one section, as
in `api.guide("python")` for the API. The shell commands are
`python -m ltspice_mcp.api reference [op]` and
`python -m ltspice_mcp.api guide [section]`.

An assistant inside a session can use the same API through `run_code`,
without installing the package.
