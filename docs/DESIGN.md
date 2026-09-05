# ltspice-mcp — Design

Scope, architecture, and the design principles you'll encounter when
using the server.

For install + client setup see [README.md](../README.md). For the parser
architecture see [docs/spice_lex.md](spice_lex.md). For current bugs and
limitations, check the GitHub issue tracker.

## Scope

The server targets **structured, validated, geometry-aware,
LTspice-specific** operations. Each adjective constrains what the
project invests in:

- **Structured** — tool outputs declare schemas; results are typed, not
  free text.
- **Validated** — mutating tools refuse invalid states before touching
  the file.
- **Geometry-aware** — `.asc` schematic operations work in symbol
  coordinates, not just netlist tokens.
- **LTspice-specific** — `.asc`/`.asy`/`.raw` behavior and Windows
  interop are handled directly. ngspice is fully supported for simulation,
  result parsing, diagnostics, and analysis; it has no schematic format, so
  the geometry tools do not apply to it. QSPICE and Xyce are best-effort
  through spicelib's common interface.

### Why an MCP server vs. asking the LLM to use spicelib directly

An LLM with code execution (Claude Code, Cursor) could write spicelib
scripts directly. MCP applies in specific contexts:

- **Works without code execution.** Claude Desktop, ChatGPT, web chat
  clients have no shell or Python. MCP tools are the only way to give
  them simulation capabilities. This is the primary audience.
- **Reliability.** `run_experiments(circuits=[{"path": "foo.cir"}])` is a
  tested code path. LLM-generated spicelib code makes import mistakes,
  calls wrong method names, forgets `save_netlist()`, etc.
- **Context efficiency.** A tool call is ~50 tokens; equivalent Python is
  20-40 lines. The difference adds up over an iterative design session.
- **Structured analysis as the default.** The `analyze_results` recipes
  return schema-typed numbers: `bode_filter` for -3 dB points;
  `bode_slope` / `bode_point` / `bode_crossing` for slopes, gains, and
  crossings; `signal_stats` / `edges` / `transient_response`
  (`mode="step"|"disturbance"`) for transient shape. The common questions
  are answered without reading anything off an image. Decimated
  raw-waveform export is the `waveform` recipe (a min/max stat-envelope for
  seeing shape); rendered plots are the `plot` recipe and the
  `plot_waveform` tool. Both complement the structured numbers for the
  shape-recognition cases a scalar can't cover (see *Waveforms: scalars
  first, export and plots for shape*); they do not replace them.
- **No user setup.** Install the server once. No spicelib in the user's
  project venv, no Python knowledge required.

Where MCP is weaker: a fixed tool set is less flexible than arbitrary
Python, and there's an extra process to maintain.

## Comparison with alternatives

|tool|schematic editing|geometry validation|cross-run analysis|WSL|
|-|-|-|-|-|
|`ltspice-mcp` (this)|direct `.asc` editing|pin coords, bbox, diagonal-wire refusal, named-net-short detection|sweep + Monte Carlo + batch metrics|supported|
|spicelib `AscEditor`|text-level mutator|none|via SimStepper / Montecarlo (lower-level)|N/A|
|PyLTSpice|spicelib re-export|none|same as spicelib|N/A|
|`xuio/ltspice-mcp`|create/modify/lint|not compared|not compared|macOS-only|
|`daviditkin/ltspice-mcp`|netlist-level (9 tools)|none|none|not compared|
|SPICEAssistant (arxiv 2507.10639)|none|N/A|measurement extractors|N/A — research only|
|LTspice GUI|interactive|interactive|GUI-driven|N/A|

"not compared" means we have not run that project and are not claiming
anything about it either way. It is not a statement that the capability is
absent — the other cells are read from each project's own description, and
none of the third-party rows have been benchmarked against this one.

Geometry-aware editing is `edit_schematic`, one transactional op batch
(`add_component`, `move_component`, `remove_component`,
`set_component_value`, `set_component_attribute`, `wire_pins`,
`add_net_label` with `pin="M3.S"`, `remove_net_label`, `remove_wire`,
`add_directive`, `remove_directive`). The ops work against pin
coordinates, bounding boxes, and named-net topology: the `wire_pins` op
refuses diagonal wires,
pin collisions, wire-junction overlaps, and named-net shorts before
touching the file, and the batch returns geometry the agent
can use in its next call. `inspect(kind="symbol")` is the read-only half
of that workflow: it returns pin positions and bounding boxes so the caller
can plan an edit. It does not validate one.

### One authoring tool, many ops

Schematic mutation is a single tool with a discriminated op union rather
than one tool per mutation. A tool's schema costs the model context whether
or not the tool is ever called, and MCP guidance is to prefer fewer, more
capable tools (tool-selection accuracy degrades past roughly 15 tools). So
the ops that return geometry (`add_component` gives placed pins, bounding
box, and overlap warnings; `wire_pins` gives the routing result) and the
ops that only acknowledge (`move_component`, `remove_component`,
`set_component_attribute`, `add_net_label`, `remove_net_label`,
`remove_wire`) all go through the same call, and a caller batches them into
one transaction. Creating a sheet is the same tool with the blank base.
Reading (`inspect`) is a separate tool, because a read-only operation
should not be hidden behind a mutating tool's schema.

## Design principles

### Validate before write

Mutating tools validate before writing and refuse states they can
prove invalid.
The `wire_pins` op is the reference example: a refusal names the specific segments,
pins, or labels that blocked the write so
the agent can pick a new waypoint instead of guessing.

Agents are bad at undoing mistakes. Tools that refuse invalid states save
entire conversation turns of recovery.

### No auto-routing

When the server refuses a wire, it returns the **conflict set**: what
blocked it and where. It does not calculate an alternate route. General
routing is NP-hard and is outside the project's scope; the caller uses the
conflict set to choose another route.

### No preview mode

There is no `dry_run` / preview parameter. Three mechanisms cover safe
mutation instead:

- **Validate-before-write refusals** — the `wire_pins` op refuses invalid
  geometry before the file is touched, with itemized error text naming
  the conflicting segments, pins, or labels; the `add_component` op
  places the part and returns its pin positions plus
  non-blocking overlap warnings.
- **All-or-nothing commits.** An `edit_schematic` batch is applied to one
  in-memory editor and written by atomic rename. `expected_sha256` names
  the revision the batch was prepared against; if another writer committed
  first, the call returns `revision_conflict` and writes nothing. A failed
  batch leaves the sheet unchanged, so recovery is re-reading the file, not
  undoing a partial write.
- **`verify_circuit` comparison** — a committed sheet is exported on a
  copy and compared against a reference netlist (equivalence or
  structural diff), so unintended drift is visible immediately.

### Post-op validation pass

Even when every op succeeds, the response includes schematic-level
warnings (floating pins, duplicate wire segments, dangling net labels).
They are cheap to compute in the same editor session and save the agent a
follow-up inspection call.

### Structured outputs

Tools that produce inspectable state declare an `output_schema` and
return `structuredContent` alongside text. When MCP clients run code
execution against tool calls, the schemas serve as the types the
sandboxed code consumes.

### Waveforms: scalars first, export and plots for shape

By default the analysis tools return scalars, not samples: time-weighted
RMS, rise time, phase margin. The consumer is an LLM, and a correct scalar
is easier to reason over than a large array of points. This default covers
waveforms whose shape a single number captures; it answers "what is X". It
does not cover cases where the shape itself is the result: switching-
converter nodes, amplifier internal nodes, startup transients.

Those cases fall into two groups with different needs:

- **Known shape question** (subharmonic oscillation, CCM/DCM mode, slew
  limiting, switch-node ringing) → a specialized detector (FFT, mode
  detector, dV/dt), not raw data: it returns the number directly instead
  of making the caller read it off a curve. Prefer a detector when one
  exists.
- **Exploratory "let me look"** (why won't this converter start?) → no
  detector helps, because the metric is not known yet; you look, form a
  hypothesis, then measure. This is the case waveform export exists for.

What matters is who consumes the output and in what format, not scalar
versus waveform:

- **LLM computing** → decimated numeric arrays, with **min/max envelope**
  decimation (each bucket carries its min and max), never stride — stride
  drops glitches and ringing peaks; the envelope preserves every
  extremum, losing only sub-bucket timing. The LLM computes on the array
  (FFT, period, overshoot), which is more reliable than reading a picture.
- **LLM recognizing shape** → a rendered plot, as a backup and complement
  to the array. A vision model sees overall shape (slewing, a settling
  envelope, "this looks unstable") at a glance, which is tedious to derive
  from numbers; plot plus array is more robust than either alone.
- **Human** → the same output handed off (CSV to a viewer, PNG inline).
  Recognition reliability is not a concern here.

So the surface is layered: scalar detectors for "what is X", specialized
shape-detectors for known signatures, decimated export for "let me look",
and plots as a shape-recognition backup for both LLM and human. Decimated
export is the `waveform` recipe; rendered plots are the `plot` recipe and
the `plot_waveform` tool.

**Scalar-guided zoom.** The waveforms this server handles (switching
nodes, amplifier internal nodes) are multi-scale: a piecewise-smooth
backbone plus sparse localized events (ringing spikes, coupling glitches,
crossover kinks). No single fixed-resolution view fits, so the caller
narrows in step by step instead of relying on one lossy encoding. The
overview is a **stat-envelope**: per bucket, `[min, max, rms, mean]`.
Min/max guarantee that a narrow spike's amplitude survives bucketing;
rms/mean localize energy and drift. From the per-bucket values the consumer
picks a sub-window and re-requests it at finer resolution (the same
`waveform` recipe with a narrower `window`), and repeats. Full-resolution
data stays on disk; detail enters context only when a measured value
justifies it. This is the same look-then-zoom loop an engineer uses on a
scope, and it reuses windowed export plus the detectors (an
envelope+carrier or edge-metric descriptor is the payload for a flagged
window at zoom time).

**Report facts without rating them — applied to navigation.** The stat-envelope and
any ranking follow the same rule as the rest of the result layer: the tool
reports measured conditions (crest factor, a bucket's spread relative to
its neighbors, `mean` drift, alternating-peak spacing, spectral
concentration) and may sort buckets by a named quantity, but it attaches
**no phenomenon label and no trust verdict**. "Crest factor 8.1 here, 5.2×
the median bucket" is a fact; "spike", "unstable", or "unreliable" is the
model's conclusion. Relative comparisons are preferred to bare-magnitude
thresholds. The model decides what the shape is and where to look next;
the server reports the facts.

**Optional always-attach plot.** A config default plus a per-call tool
parameter make waveform / analysis responses include a rendered plot (MCP
`ImageContent`) of the queried window in addition to the scalars and the
stat-envelope data points, not instead of them. The plot is the
shape-recognition layer for the vision case above. Because it renders the
queried window, the same zoom loop re-renders it at finer scale. Off by
default (token cost, non-vision clients); on for clients that want a plot
with every result; a per-call override is available either way.

### Export & plot surface (as of 2026-06-13)

The layered surface above maps to three delivery channels, one per
consumer. The channel for each is fixed even though one library choice is
still open.

| Consumer | Needs | Channel |
|-|-|-|
| The agent (computes) | the numbers, full fidelity | the `waveform` recipe in `format: "csv"` → CSV on disk |
| A vision model (one frame) | shape-at-a-glance | static PNG (`ImageContent`) attached to a result |
| A human (explores) | an interactive plot | `plot_waveform`, rendered to the richest surface the client supports |

One constraint affects the last two: the terminal CLIs do not render images
for the human. Claude Code passes an `ImageContent` PNG to the model (vision
works) but never shows it to the user; Codex handles tool-returned images
unreliably (it tends to dump base64 into context). The MCP "apps" widget
surface (`ui://` HTML in a sandboxed iframe) is available only on GUI hosts
(Claude Desktop / claude.ai web), not in the terminal. And interactivity
benefits the human, not the LLM: a model consumes a single rendered frame,
so zoom / pan / hover does nothing for it.

- **The `waveform` recipe in `format: "csv"` — full-fidelity export to disk.**
  Returns a path, not data; full resolution in a response would exceed the
  context budget, which is why the inline form decimates. Works on every
  analysis type. `.step` / Monte-Carlo runs are written in long (tidy) form
  (`step_index, step_value, x, <signals…>`) because transient `.step` runs
  have a different time vector per step, so a wide shared-`x` layout would be
  wrong. Complex AC traces are written as magnitude (dB) + phase (deg)
  columns: lossless, plot-ready, and the form the AC structural-analysis
  functions read. This is the only path that emits every sample of the
  complex `H(f)` array (the inline form decimates and the Bode recipes return
  scalars), so the AC structural-analysis layer is built on it. It exports
  raw data only: no derived slope / group-delay / residual columns (those
  belong to the detectors). Size is bounded by windowing (the zoom loop
  above), not by decimation. No new dependencies. Column scheme: a
  unit-tagged x header (`time_s` / `freq_Hz` / `sweep`); real traces under
  their canonical name (`V(out)`); complex traces as `V(out)_mag_dB` +
  `V(out)_phase_deg` (or `_re`/`_im`, or all four). Phase is the wrapped
  `np.angle`; a consumer runs `np.unwrap` itself. Non-finite samples are kept
  and counted, not dropped, so columns stay row-aligned. The CSV is written
  under `<working_dir>/.ltspice-mcp/results/artifacts/<result_set_id>/`
  (Linux-side, never beside a Windows-temp raw). A descending or
  non-monotonic axis is refused when windowed (searchsorted would corrupt
  it). Rows stream straight into the atomic temp file (no whole-CSV copy held
  in memory) under a generous row-count limit that raises rather than
  truncates; the resolved output path is rejected if a symlinked sidecar
  would redirect it outside the circuit directory; and in a stepped run, a
  step whose axis misses the window is skipped and reported rather than
  failing the whole export.
- **`plot_waveform` — one chart core, two delivery paths.** One interactive
  chart, delivered one of two ways depending on the client detected at
  `initialize`: an in-chat `ui://` widget for an apps-capable GUI host (in
  practice Claude Desktop for a local stdio server), or a self-contained HTML
  file opened on the local desktop for a terminal client (on WSL,
  `explorer.exe` / `cmd.exe /c start` via a `wslpath -w` conversion;
  otherwise `xdg-open` / `open` / `start`). Either way the tool also returns
  a text summary and the data path so the model can continue. Because the
  server is local, an interactive plot needs no widget infrastructure: it
  opens a window or browser that the OS renders, which works the same under
  any CLI. The chart is built on uPlot (~50 KB, zero-dependency, canvas-2D)
  inlined into one self-contained HTML file. It inlines cleanly into both
  the offline file and the payload-capped widget and can render
  full-fidelity transients; Plotly (~1.5 MB, heavy to inline per widget) lost
  on both inline size and rendering performance with full data. It is not a
  Python plotting dependency. Fidelity: full by default, a `max_points`
  parameter to override, and a min/max-preserving downsample that engages
  only above a high cap and is reported in `observations` (no silent
  truncation). Both delivery paths are implemented. uPlot is vendored as a
  bundled MIT asset under `src/ltspice_mcp/assets/uplot/` (in the wheel,
  inlined at render; no pip dependency, no CDN). A step whose axis misses the
  window is skipped; transient `.step` overlays with differing per-step time
  vectors are null-padded onto a union x; AC Bode phase is unwrapped;
  non-finite samples become JSON `null` gaps. A global per-panel cell cap
  refuses, before allocating, a plot whose union-padded size would be too
  large. Delivery is chosen by the client detected at `initialize`
  (`capabilities.extensions["io.modelcontextprotocol/ui"]`):
  - **MCP Apps host (SEP-1865, Final 2026-01-26)** → an in-chat `ui://`
    widget, wired the standard way rather than by inline embedding: the
    `plot_waveform` tool declares `_meta.ui.resourceUri`; one stable,
    predeclared renderer resource (`ui://ltspice-mcp/plot`, uPlot plus the
    vendored Apache-2.0 ext-apps `App` runtime inlined) is served via
    `resources/read`; the host renders it in a sandboxed iframe and passes it
    a compact chart spec (decimated harder than the file, carried in the
    result `_meta`, a channel the model does not see, so the plot still
    returns no numbers to the model), which `app.ontoolresult` draws. The
    full-fidelity HTML still lands on disk; an oversized spec (byte cap) or a
    build failure falls back to local-open and is reported.
  - **Terminal client** → the self-contained HTML is opened locally
    (`explorer.exe` via `wslpath -w`, else `xdg-open`/`open`/`startfile`),
    spawned detached so it never blocks.
  Both always return the file path and a text summary, so a host with neither
  surface still gets a usable result (the fallback the MCP Apps spec
  describes).
- **Static PNG (the vision tier) — opt-in.** A config default
  `[analysis] attach_plot` (off) plus a per-call `attach_plot` tool
  parameter that overrides it: an operator can attach a plot to every
  analysis result, and the model can opt in or out for a single call.
  Default off (a base64 PNG on every result is expensive, useless to
  non-vision clients, and barely works in Codex). Gated on the optional
  `[plot]` extra (matplotlib); if it is absent, skip and report rather than
  error. Scoped first to the recipes where a plot helps most (`waveform` and
  the Bode recipes). Not built; the interactive channel covered the need
  first.

**Fidelity by consumer.** The inline `waveform` recipe decimates (the LLM's
context is the limit), its `format: "csv"` form is lossless (disk has no such
limit), and
`plot_waveform` defaults to full fidelity (a browser is not context-bound),
capping only at very large sizes.

**Dependencies.** The core install adds no plotting dependency. The
interactive channel adds only uPlot (~50 KB, zero-dep, inlined); matplotlib
is confined to the static-PNG tier and ships as the optional `[plot]` extra.
**Renderer principle:** the plot layer takes plain arrays plus labels and
knows nothing about job or run internals, so the three channels stay
swappable behind one data contract.

## Architecture

```
MCP protocol layer    server.py — lifespan, dispatch, request routing
                      resources.py — MCP resources & URI templates
Tool layer            tools/*.py — tool definitions + handlers
Core logic            lib/*.py
Config / state        config.py, state.py, errors.py
```

Tool modules use a decorator-based registry (`@registry.tool(...)`).
Each registration declares `name`, `description`, `input_model` (Pydantic),
`annotations`, and an optional `output_schema`. `SessionState.create()`
builds the surface once during lifespan init; the dispatch table and the
tool definitions both come from `registry`.

Key `lib/` modules:

|module|purpose|
|-|-|
|`services.py`|service layer shared by tools and resources — job resolution, cached result loading, reusable extraction|
|`runner_base.py`, `experiment_runner.py`|the spicelib runner wrapper and the experiment coordinator built on it|
|`schematic_ops.py`|the `.asc` edit engine — op models, appliers, geometry, net tracing|
|`runner_manager.py`|centralized runner lifecycle with auto-invalidation on loop / simulator / output-folder change|
|`simulator.py`|simulator detection, WSL/Wine selection|
|`ltspice_wsl.py`, `wsl.py`|WSL path conversion and Windows interop|
|`raw_parser.py`, `log_parser.py`|simulation result parsing|
|`library_manager.py`, `library_parser.py`|component library handling|
|`symbol_geometry.py`|`.asy` symbol parsing, pin positions, rotation transforms, bounding boxes|
|`spice_lex.py`, `spice_lex_ops.py`, `spice_lex_views.py`|shared SPICE lexer pipeline — see [docs/spice_lex.md](spice_lex.md)|
|`pathutil.py`|path security (`safe_path()`, `resolve_safe_path()`)|

### The tool surface

`config.tool_profile` controls which tools are exposed. Since 0.6.0 there
is one profile.

|profile|tool count|use case|
|-|-|-|
|`consolidated` (the only profile)|7 tools|Any MCP client: `run_experiments`/`jobs` (execute), `analyze_results`/`inspect` (understand), `edit_schematic`/`verify_circuit` (author), plus the `plot_waveform` widget|

Each of the six tools takes a declarative payload rather than a fixed
argument list, so the number of capabilities did not shrink with the number
of tools: sweeps, corners and Monte Carlo are `run_experiments`
`variations`; every former analysis tool is an `analyze_results` recipe;
every former schematic mutation is an `edit_schematic` op; the former read
tools are `inspect` kinds. Netlist text editing has no tool at all: an
agent with file access does it natively, and a wrapper added nothing.

**What 0.6.0 removed.** The `full` (49-tool) and `agentic` (41-tool)
profiles, and with them one tool per operation, the single-simulation and
batch job types those tools ran, and the runners behind them. A job
sidecar an earlier release wrote still loads, inert: reading one reports
what it is and that it must be re-run. Their profile names are still
accepted in `[tools] profile` and `LTSPICE_MCP_TOOL_PROFILE` for one
release; each logs a warning and serves the consolidated surface, so a
config that names a removed profile still starts a working server instead
of failing. A deployment that needs the old tools pins
`ltspice-mcp==0.5.*`. The handlers behind those tools were kept as internal
adapters through 0.6 development, so the consolidated surface could be shown
to run the same code paths; they are gone now, and the numeric core they
carried lives in `lib/metrics.py` as one function per recipe.

## Backend: spicelib

`spicelib` (>= 1.4.9, < 1.6) is the core Python library for SPICE
automation, by Nuno Brum. PyLTSpice is a thin re-export wrapper over
spicelib that adds nothing — we depend on `spicelib` directly. The
ceiling is deliberate: spicelib 1.6 retypes `.PARAM` values (`float_unit`)
and refactored the component model, which breaks the parameter and
structural-comparison paths. See the pin comment in `pyproject.toml`. Raise
the bound only when something forces it, not to pick up features.

All four simulators share the same base `Simulator` ABC:

|simulator|class|platform|
|-|-|-|
|LTspice|`LTspice`|Windows native; Linux/macOS via Wine; WSL via Windows interop|
|NGspice|`NGspiceSimulator`|Linux/macOS/Windows native|
|QSPICE|`Qspice`|Windows; Wine limited|
|Xyce|`XyceSimulator`|Linux/Windows native|

Core spicelib components used:

|component|purpose|
|-|-|
|`SpiceEditor`|read/modify/write `.net`/`.cir` netlists|
|`AscEditor`|read/modify `.asc` schematics (LTspice)|
|`SimRunner`|batch execution with parallel-sim support|
|`RawRead`|parse binary `.raw`/`.qraw` waveform output (dialect auto-detection)|
|`LTSpiceLogReader`|extract `.MEAS`, step data, Fourier|
|`SimStepper`|multi-dimensional parameter sweeps (overcomes the 3-parameter `.STEP` limit)|

Monte Carlo is not spicelib's `Montecarlo` toolkit: perturbation lives
in the in-repo pure engine (`lib/montecarlo.py`) and runs through the
same `SimRunner`-based execution path as everything else.

Per-simulator notes:

- **LTspice** supports `.asc` → `.net` conversion via spicelib's
  `LTspice.create_netlist()`.
  macOS LTspice has no CLI switch support. Default switches: `-Run -b`.
- **NGspice** has a compatibility mode (`kiltpsa` default for
  KiCad/LTspice/PSPICE), overridable via `[simulator] ngbehavior` (or the
  `LTSPICE_MCP_NGBEHAVIOR` env var) — e.g. `hsa` when a sectioned
  `.lib <file> <section>` corner select must survive, since `kiltpsa`'s
  `lt`/`ps` tokens make ngspice read it as two plain includes and drop the
  section. Default switches: `-b -o -r -a`. Native on Linux.
- **QSPICE** uses `.qraw` (double precision). Windows-only, limited Wine.
- **Xyce** supports `-syntax` and `-norun` for validation without
  simulation.

## WSL support

On WSL, LTspice runs via Windows interop, not Wine. Operational
adaptations:

- **Path conversion**: `LTspiceWSL` subclass (`lib/ltspice_wsl.py`)
  overrides `run()` to convert paths via `wslpath` instead of Wine's
  `Z:` prefix. Auto-selected when `is_wsl()` is True.
- **`%LOCALAPPDATA%` resolution**: symbol-library paths under
  `%LOCALAPPDATA%` are resolved via `cmd.exe` since they vary by
  Windows user profile.
- **Windows-side output dir**: simulation output goes to a Windows-native
  temp dir when the working dir is on the Linux filesystem. Required for
  `.MEAS` results — LTspice's SQLite `.db` writes fail on UNC paths
  (`\\wsl.localhost\...`), which loses measurement data from `.log`
  files.
- **Extension preservation**: LTspice requires netlist files to have an
  extension (`.cir`, `.net`, `.sp`). `runner_base.submit_netlist` preserves
  the original extension in `run_filename`.

`simulator.path` in `ltspice-mcp.toml` must point at the
Windows-side executable (e.g.
`/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) — spicelib can't
auto-detect across the WSL boundary.

## Diagnostics

`log_parser.py:extract_log_diagnostics()` extracts structured warnings
and errors from simulator log files — parse errors with caret pointers,
fatal errors, convergence messages, `.MEAS` parse failures — and
`run_experiments`, `jobs`, and the `summary` recipe attach them to
their responses as `warnings` / `errors` lists. An agent driving raw
`ngspice -b` gets a 200-line log dump and has to grep; here the same
failure arrives inside the tool response.

Above the raw lists, the observation surfacer
(`lib/result_observations.py`) adds an `observations` list to the run
summary with the facts most likely to matter:

|observation kind|what it surfaces|
|-|-|
|`relay`|the simulator's own error lines, with the simulator's severity (never an invented one)|
|`reconciliation`|requested `.meas`/`.four` outputs that were not produced|
|`value`|non-finite samples and extreme node values in the trace data|
|`coverage`|checks that were skipped, so a thin result can't pass as a clean one|

The surfacer does not rate the result; it reports facts for the model to
judge. An empty `observations` list means nothing tripped a check, not that
the result is verified correct.

**Scope:** LTspice and ngspice both get the full pipeline; ngspice's stdout
diagnostics are captured alongside its log file and fed through it. qspice
/ xyce get best-effort convergence and fatal-error parsing.

## Non-goals

These are intentional gaps, not pending features.

- **Auto-routing / auto-placement.** Conflict-set returns on refusal
  are the substitute. General routing is out of scope.
- **Semantic part search** ("find me a low-Vgs-th NMOS under 100mΩ
  Rds(on)"). Requires parsing every `.model` card, normalizing units
  across vendors, building a queryable parameter index — a separate
  parameter-database effort.
- **Thin wrappers around file-edit operations.** Netlist text editing has
  no tool: a client with native `Read`/`Edit` does it better, and one
  without it can still hand the server a deck it wrote.
- **Per-simulator parity for geometry.** LTspice and ngspice are
  co-equal for simulation, result parsing, diagnostics, and analysis.
  The geometry layer (`.asc`, `.asy`, symbol coordinates) stays
  LTspice-only by nature — ngspice has no schematic format to edit.
  QSPICE and Xyce get best-effort support through spicelib's common
  interface.
- **Auth / multi-tenancy / remote-MCP plumbing.** This is a local tool:
  no OAuth flows, no per-tenant isolation.
- **Generic cross-simulator EDA features** that PyLTSpice and ngspice's
  CLI already cover well.

## Roadmap

Forward-looking and **not commitments**. Priorities follow the scope
statement at the top of this doc: work that is structured, validated,
geometry-aware, and LTspice-specific comes first.

- **Layout primitives**: `route_bus`, `align`, `distribute` for the
  rows of caps/resistors agents constantly need to place.
- **`ltspice-mcp doctor`**: one-shot diagnostic that checks simulator
  detection, symbol-path resolution, output-dir placement, Wine vs WSL
  selection, `.MEAS`-on-UNC risk. The `flutter doctor` / `brew doctor`
  pattern.
- **Diagnostic taxonomy expansion**: singular matrix, gmin stepping
  suggestions, timestep too small, missing model cards, unmatched
  subcircuit pins, `.STEP` parameter mismatches. Each gets a
  `suggested_fix` field. LTspice- and ngspice-scoped.
- **Cross-run analysis**: `compare_corners`, `find_worst_case`,
  `sensitivity_ranking` — tools that aggregate measurements across a
  set of runs and return structured deltas.
- **Waveform export & plotting**: shipped (see *Export & plot surface*
  above) — the `waveform` recipe's CSV form and both `plot_waveform`
  delivery tiers. What remains open is the opt-in static-PNG attach for the
  vision tier.
- **Pin-compatible alternate suggestions** for unknown parts.

## Configuration

`ltspice-mcp.toml` in the working directory (auto-generated if missing).
Environment variables with `LTSPICE_MCP_` prefix override TOML values.
See `config.py:ServerConfig` for all options.

TOML sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`,
`[logging]`, `[schematic]`, `[tools]`, `[state]`.

## Verification recipe

End-to-end smoke test for a fresh install:

1. Create RC lowpass netlist (R=1k, C=100n → fc ~1.59kHz)
2. Add `.ac dec 100 1 1Meg` directive
3. Run it with `run_experiments`
4. Read it back with a `bode_filter` recipe — verify the -3dB point near
   1.59kHz
5. Change R to 10k (fc → ~159Hz)
6. Re-run and re-measure — verify the bandwidth shifted
7. Pull a custom library in with a `.lib` / `.include` directive and use one
   of its models in a new circuit

If a step fails, check `simulator.path` in TOML and confirm
`ltspice-mcp.toml`'s `[security] allowed_paths` includes your working
directory. The planned doctor tool (see Roadmap) will automate these
checks.

For how the project tests, see `docs/TESTING.md`. It also describes the
absence-class bugs the tests once missed and the mechanisms added to catch
them (inverse-op closure, the archetype build battery, task-down coverage,
blind-artifact judging).
