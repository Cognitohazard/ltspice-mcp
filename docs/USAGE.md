# Using ltspice-mcp

Details the [README](../README.md) leaves out: setup options, the LTspice
window features, configuration, and the Python API.

- [Installing](#installing)
- [Making sure the assistant uses the server](#making-sure-the-assistant-uses-the-server)
- [Working with LTspice open](#working-with-ltspice-open)
- [Alongside LTspice's own MCP server](#alongside-ltspices-own-mcp-server)
- [Configuration](#configuration)
- [Files the server creates](#files-the-server-creates)
- [WSL](#wsl)
- [PNG rendering](#png-rendering)
- [Where the server can run](#where-the-server-can-run)
- [What a session looks like](#what-a-session-looks-like)
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
assistant may run the simulator from the shell instead. Two things help.

**The server's name.** The assistant sees every tool prefixed with it, as in
`mcp__spice__run_experiments`, so a name that says SPICE helps. It is the key
in your client's MCP config, or the word after `claude mcp add`. The plugin
uses `spice`, and the Desktop extension is listed under its own name. If
yours is something generic like `sim1`, rename it. Rename `ltspice` too,
because LTspice's own MCP server uses that name.

**An instruction.** Put the rule from the README in your project's
`CLAUDE.md`, or whatever your client calls its project instructions. The rule
is absolute on purpose: an assistant told to weigh the choice usually picks
the shell it already knows.

### When the shell is fine

If you drop the rule, this is the real boundary. An assistant with a shell
can run quick, one-off ngspice simulations directly; they are scriptable and
usually take under a second, so the server adds little. Use the server for
LTspice runs, values read by name from binary `.raw` files, sweeps and Monte
Carlo, jobs that outlive one call, and checked `.asc` editing.
`analyze_results` can also measure a `.raw` file produced outside the server,
so a simulation run from the shell can still be analyzed here.

## Working with LTspice open

These features need Windows and LTspice 26.1 or later. The server reaches
your LTspice window through `ltspice-mcp-bridge.exe`, a program LTspice
installs.

**Edits to an open schematic.** LTspice does not reload a schematic that
changes on disk, so without this the window would keep showing the old
version, and saving it would undo the assistant's edit. Instead, when the
assistant edits a schematic you have open, the server updates the window as
well: the change appears at once, and Ctrl+Z undoes it. If the window has
unsaved changes, the edit is refused and the assistant asks you to save or
close the schematic first. Editing never starts LTspice; it only updates a
window that is already open. `[schematic] sync_open_window = false` turns
this off.

**Showing you things in LTspice.** The assistant can see which schematic you
have in front, so "this circuit" means something. When you ask, it can open a
schematic in LTspice, or open a simulation run with the traces you asked
about already plotted. If LTspice isn't running, it starts it for these two
requests only; `[schematic] start_ltspice = false` turns that off.

When the run is of a schematic, clicking a net in the window plots it, as if
you had run the schematic in LTspice. To make that work, the server copies
the run's results next to the schematic as `<name>.raw`, replacing any file
of that name, and opens them from the schematic. It doesn't simulate again.

**LTspice's documentation.** For questions about LTspice itself, such as a
shortcut or a menu, the assistant reads the reference files LTspice 26.1 and
later installs.

None of this needs LTspice's own MCP server.

## Alongside LTspice's own MCP server

LTspice 26.1 and later ships its own MCP server. On Windows it offers to add
itself to Claude Code, Claude Desktop, Copilot and Cursor as `ltspice`.

The two do different jobs, and you can install both. LTspice's server
controls the LTspice window: it can read a schematic you haven't saved and
run it on screen while you watch. It returns raw samples and the log, so
working out a phase margin or setting up a Monte Carlo run is left to the
assistant, and it can only edit a schematic by replacing its whole text. This
server works on the files instead: it places and wires parts with the
geometry checked, runs sweeps and Monte Carlo as background jobs, returns
measurements as numbers, and also runs ngspice.

**A setting worth changing on LTspice's side.** When LTspice's server has no
window to use, it starts a hidden copy of LTspice, one per assistant session,
and changes made through it can end up in a copy you can't see. To prevent
that, register it with `--ltspice-path` set to a file that doesn't exist; it
can then only use a window you have open. This is a workaround, not a
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
# allowed_paths = ["."]  # folders the server may use; unset = the working directory and the Claude Code scratch folder

[simulation]
# max_parallel = 4       # default: number of CPU cores, capped at 8
timeout = 300.0          # seconds, for LTspice netlist export
# run_timeout = 3600     # seconds per run when a request sets none; default: no limit

[schematic]
sync_open_window = true  # Windows, LTspice 26.1+: show edits in the LTspice window that has the schematic open
start_ltspice = true     # start LTspice when asked to show something in it and it isn't running

[tools]
listing = "compact"      # "full" sends every argument description, about 45% more to load
run_code = true          # false removes run_code

[state]
persist_jobs = true
```

**`listing`.** The default, `compact`, leaves argument descriptions out of
the tool list, which cuts about 45% from what a session loads before its
first call. The tools accept exactly the same calls. The assistant looks up
an argument with `inspect(kind="reference", query="...")`, and a rejected
call lists the fields it accepts. `full` sends every description.

**`run_code`** runs a Python snippet in a worker process that holds the
engine as `api`, for loops over runs and numpy on samples. The snippet runs
with the server's own file and process permissions, not inside
`allowed_paths`. Approve `mcp__spice__run_code` in your client as you would a
shell, and don't allow it as part of a blanket `mcp__spice__*` rule. Set
`run_code = false` if more than one client can reach the server, for example
through a proxy. The change takes effect at the next start;
`inspect(kind="capabilities")` reports whether the tool is on.

**Plots.** `plot_waveform` opens its chart in a browser when the client can't
show it in the chat; `[analysis] open_plot = false` returns only the file
path. `[analysis] attach_plot = true` also returns a PNG of the chart for the
assistant to look at (about a thousand tokens per call), which needs
[PNG rendering](#png-rendering).

## Files the server creates

Beside your circuits, the server writes only files you ask for (an edited
schematic, an exported netlist, a plot saved to a folder you name), plus:

- the `<name>.net` that LTspice writes next to a schematic it runs;
- the `<name>.raw` placed next to a schematic when you ask to see its run in
  LTspice.

Everything else goes in two places:

- **`.ltspice-mcp/` in the working directory:** job records, runs, results,
  export snapshots and plots. Add it to `.gitignore`.
- **A per-user folder:** the recent-circuits list and lock files that keep
  two sessions from editing one file at once. It is
  `%LOCALAPPDATA%\ltspice-mcp` on Windows and `~/.local/state/ltspice-mcp`
  (or under `$XDG_STATE_HOME`) elsewhere; `LTSPICE_MCP_HOME` moves it.

Two environment settings reduce what lands in the working directory:

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

LTspice's symbol libraries, needed for `.asc` editing, are found
automatically on Windows and WSL. To use others, set
`[schematic] symbol_paths` or `LTSPICE_MCP_SYMBOL_PATHS`. A symbol saved in
the same folder as the schematic is always found first, as in LTspice.

## PNG rendering

`verify_circuit` draws schematics as SVG, but an assistant can only see a
drawing returned as a PNG, since clients don't reliably display SVG. PNG
needs two things the plain install leaves out:

1. **The `raster` extra.** Install the server with it:

   ```bash
   uv tool install 'ltspice-mcp[raster]'    # or: pipx install 'ltspice-mcp[raster]'
   ```

   With `uvx`, use `uvx --from 'ltspice-mcp[raster]' ltspice-mcp`. The Claude
   Code plugin and the Claude Desktop extension already include it.

2. **The Cairo library**, which no Python package ships:
   - **Linux and WSL:** `sudo apt install libcairo2` (Fedora:
     `sudo dnf install cairo`). On WSL, install it inside the Linux
     distribution.
   - **macOS:** `brew install cairo`. On Apple silicon, also set
     `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` in the server's
     environment.
   - **Windows:** install a Cairo runtime, for example the
     [GTK for Windows runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer).
     Then add the folder containing `libcairo-2.dll` (the runtime's `bin`
     folder) to `PATH`, or name it in `CAIROCFFI_DLL_DIRECTORIES` (separate
     several folders with `;`).

Restart the server afterwards. The image `plot_waveform` attaches with
`attach_plot` needs the same two things.

If either is missing, a PNG request returns an SVG file instead, and the
reply names what is missing and how to install it; a plot still returns its
chart and trace summaries. `inspect(kind="capabilities")` reports the same
before anything is drawn:

```json
"render": {"png": false, "missing": "native_library", "reason": "...", "remedy": "..."}
```

`missing` is `"extra"` or `"native_library"`; it, `reason` and `remedy` are
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
on a network you fully control: it writes files and starts processes inside
`allowed_paths`. [SECURITY.md](../SECURITY.md) has the threat model.

## What a session looks like

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
`jobs(action="status" | "wait" | "cancel")` follows it. Jobs, signals,
measurements and the configuration are also readable as MCP resources
(`spice://results/...`, `spice://netlists/...`, `spice://config`).

Sweeps and Monte Carlo are `variations` of `run_experiments`, not separate
tools:

- `assign` sets values: a `.param`, a component's value (so a supply voltage
  is an assignment to its source), `REF@model` to swap one device's model, or
  `X1:delvto` for an offset on a transistor inside a subcircuit. Several
  entries form a grid; lists within one entry step together. Swapping models
  is how corners are run.
- `random` adds Monte Carlo runs: component tolerances, `.param` and
  `.MODEL` parameter spread, and Pelgrom W·L device mismatch.

Temperature is set in the deck (`.temp`, `.step temp`, `.options temp=`),
not as a variation. A `.param TEMP` is rejected, because SPICE never uses it
as the simulation temperature.

Every measurement, `inspect` kind, schematic operation and check is listed
by `inspect(kind="reference")`, generated from the tools themselves. The
design of each tool is in
[design/mcp_surface.md](design/mcp_surface.md).

## Python API

The [README](../README.md#python-api) has a first example. The full contract
is [design/python_api.md](design/python_api.md); this section covers how the
API relates to the server.

| | MCP server | Python API |
|-|-|-|
| Called by | an assistant in an MCP client | a script, notebook or CI job, often one an assistant wrote |
| Results | large results come in pages, continued with a cursor | complete results, waveforms as numpy arrays |
| Long runs | the server keeps the job running; follow it with `jobs` | the script owns its jobs, unless it detaches them |
| Good for | interactive work: explore, edit, run a few checks | optimizers, custom post-processing, pipelines |

A sweep or Monte Carlo set is one call either way. Use the API when each run
depends on code that processes the previous result.

**Both at once.** The server and scripts work in the same directory on the
same job records, so either can read a job the other started by its
`job_id`. An assistant can start a sweep over MCP and a script can read the
finished job, or a script can run a batch for an assistant to analyze later.

**Jobs a script starts.** `api.close()`, the end of a `with` block, or normal
interpreter exit cancels the script's unfinished jobs.
`run_experiments(wait=False, detach=True)` instead hands the job to a
separate process that supervises it: the call returns once the job is
recorded, the receipt names that process and its log, and the job keeps
running after the script exits. Read or cancel it later by `job_id`, from a
script or the server.

**Differences from the tools:**

- Results are complete: the API collects every page. It rejects MCP-only
  controls such as response budgets, page cursors and wait times rather than
  ignoring them, so the same call means the same thing through both.
- One engine per process: creating an `Api` inside a running server process
  raises an error. A new `Api` starts in well under a second.
- `api.load_raw()` and `api.measurements()` return numpy data for your own
  processing.

**Finding your way.** `api.reference()` lists the six operations, and
`api.reference("run_experiments")` prints one operation's arguments.
`api.guide()` returns the guide, and `api.guide("python")` its section on the
API. From a shell: `python -m ltspice_mcp.api reference [op]` and
`python -m ltspice_mcp.api guide [section]`.

An assistant inside a session can use the same API through `run_code`,
without installing the package.
