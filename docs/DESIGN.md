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
- **LTspice-specific** — `.asc`/`.asy`/`.raw` quirks and Windows
  interop are first-class. ngspice is also first-class for simulation,
  result parsing, diagnostics, and analysis — it simply has no
  schematic layer for the geometry tools to operate on. QSPICE and
  Xyce are best-effort through spicelib's common interface.

### Why an MCP server vs. asking the LLM to use spicelib directly

An LLM with code execution (Claude Code, Cursor) could write spicelib
scripts directly. MCP applies in specific contexts:

- **Works without code execution.** Claude Desktop, ChatGPT, web chat
  clients have no shell or Python. MCP tools are the only way to give
  them simulation capabilities. This is the primary audience.
- **Reliability.** `run_simulation(netlist="foo.cir")` is a tested code
  path. LLM-generated spicelib code makes import mistakes, calls wrong
  method names, forgets `save_netlist()`, etc.
- **Context efficiency.** A tool call is ~50 tokens; equivalent Python is
  20-40 lines. Over an iterative design session this compounds.
- **Structured analysis as the default.** The analysis tools return the
  numbers an agent reasons over — `bode_metrics` for -3 dB points,
  slopes, and crossings; `signal_stats` / `edge_metrics` /
  `pulse_response` for transient shape — as schema-typed results, so the
  common questions are answered without reading anything off an image.
  Decimated raw-waveform egress ships as `get_waveform` (a min/max
  stat-envelope for seeing shape); rendered plots remain roadmapped. Both
  complement the structured numbers for the shape-recognition cases a
  scalar can't cover (see *Waveforms: scalars first, egress and plots for
  shape*), rather than replacing them.
- **No user setup.** Install the server once. No spicelib in the user's
  project venv, no Python knowledge required.

Where MCP is weaker: a fixed tool set is less flexible than arbitrary
Python, and there's an extra process to maintain.

## Comparison with alternatives

|tool|schematic editing|geometry validation|cross-run analysis|WSL|
|-|-|-|-|-|
|`ltspice-mcp` (this)|`.asc` first-class|pin coords, bbox, diagonal-wire refusal, named-net-short detection|sweep + Monte Carlo + batch metrics|first-class|
|spicelib `AscEditor`|text-level mutator|none|via SimStepper / Montecarlo (lower-level)|N/A|
|PyLTSpice|spicelib re-export|none|same as spicelib|N/A|
|`xuio/ltspice-mcp`|create/modify/lint|not documented|not documented|macOS-only|
|`daviditkin/ltspice-mcp`|netlist-level (9 tools)|none|none|not documented|
|SPICEAssistant (arxiv 2507.10639)|none|N/A|measurement extractors|N/A — research only|
|LTspice GUI|interactive|interactive|GUI-driven|N/A|

Geometry-aware editing tools — `connect`, `add_component`,
`apply_schematic_ops` (whose ops include `move_component`,
`add_net_label` with `pin="M3.S"`, `remove_net_label`, and
`remove_wire`) — work against pin coordinates, bounding
boxes, and named-net topology: `connect` refuses diagonal wires,
pin collisions, wire-junction overlaps, and named-net shorts before
touching the file, and every standalone tool returns geometry the agent
can use in its next call. `symbol_info` and `component_info` are the
read-only planning half of that workflow — pin positions and bounding
boxes looked up before an edit, not validators of one.

### Standalone tool vs. apply_schematic_ops op

A schematic mutation earns a standalone MCP tool only when its result
returns information the model acts on (structured output the next call
consumes — `add_component`'s pin geometry and bounding box, `connect`'s
routing result). An ack-only mutation — one that just confirms "done" —
lives only as an `apply_schematic_ops` op: a standalone tool's schema
costs the model context whether or not it is ever called, and that cost
is only earned by a useful return. The MCP guidance is fewer, more
capable tools (tool-selection accuracy degrades past roughly 15 tools),
so `move_component`, `remove_component`, `set_component_attribute`,
`add_net_label`, `remove_net_label`, and `remove_wire` are ops on
`apply_schematic_ops` rather than standalone tools. `create_schematic`
and `reset_schematic` are the lifecycle exception — they stay standalone
regardless of return shape because they have no batch home; reads stay
standalone too.

## Design principles

### Validate before write

Mutating tools validate before writing and refuse states they can
prove invalid.
`connect` is the model: a refusal raises an error whose itemized text
names the specific segments, pins, or labels that blocked the write so
the agent can pick a new waypoint instead of guessing.

Agents are bad at undoing mistakes. Tools that refuse invalid states save
entire conversation turns of recovery.

### Refusal, not auto-routing

When a wire is refused, the server returns the **conflict set** — what
blocked it and where — but does **not** try to auto-route around the
obstacle. Routing is NP-hard; the conflict-set form is almost as useful
at a fraction of the cost, and keeps the agent in control of the
decision.

### Recovery over preview

There is no `dry_run` / preview parameter. Safe mutation rests on three
shipped mechanisms instead:

- **Validate-before-write refusals** — `connect` refuses invalid
  geometry before the file is touched, with itemized error text naming
  the conflicting segments, pins, or labels; `add_component` places the
  part and returns its pin positions plus non-blocking overlap warnings.
- **`reset_schematic`** — reverts an `.asc` to the byte snapshot taken
  before its first in-session mutation; the recovery hatch when an
  edit sequence went wrong.
- **`export_netlist` diffing** — each export reports the diff against
  the previous export, so unintended drift is visible immediately.

### Post-op validation pass

Even when a single op succeeds, schematic-level warnings (floating pins,
duplicate wire segments, dangling net labels) ride along in the
response. Cheap to compute during the same editor session, and saves
the agent a follow-up inspection turn.

### Structured outputs

Tools that produce inspectable state declare an `output_schema` and
return `structuredContent` alongside text. When MCP clients run code
execution against tool calls, the schemas serve as the types the
sandboxed code consumes.

### Waveforms: scalars first, egress and plots for shape

Analysis tools return **scalars, not samples** by default — time-weighted
RMS, rise time, phase margin, the salient number — because the consumer
is an LLM reasoning over meaning, and a correct scalar beats a wall of
points it must interpret. This is a **scoped default, not the end state**:
it answers "what is X" for waveforms whose shape a single number
captures. It does not serve cases where the *shape itself* is the
signal — switching-converter nodes, amplifier internal nodes, startup
transients.

Those split in two, and want different things:

- **Known shape-question** (subharmonic oscillation, CCM/DCM mode, slew
  limiting, switch-node ringing) → a *specialized detector* (FFT, mode
  detector, dV/dt), not raw data: it returns the precise number instead
  of making the caller read it off a curve. Doctrine favors detectors.
- **Exploratory "let me look"** (why won't this converter start?) → no
  detector helps, because the metric isn't known yet; you look,
  hypothesize, then measure. This is the irreducible case for waveform
  egress.

The axis that matters is **consumer × format**, not scalar-vs-waveform:

- **LLM computing** → decimated numeric arrays, with **min/max envelope**
  decimation (each bucket carries its min and max), never stride — stride
  drops glitches and ringing peaks; the envelope preserves every
  extremum, losing only sub-bucket timing. The LLM computes on the array
  (FFT, period, overshoot), which is more reliable than reading a picture.
- **LLM recognizing shape** → a rendered plot, as a backup and complement
  to the array. Vision catches gestalt — slewing, a settling envelope,
  "this looks unstable" — at a glance that's tedious to derive from
  numbers; plot-plus-array is more robust than either alone.
- **Human** → the same egress as a hand-off (CSV to a viewer, PNG
  inline). Recognition reliability is moot; the human's eyes are the
  recognizer.

So the surface is layered: scalar detectors for "what is X", specialized
shape-detectors for known signatures, decimated egress for "let me look",
and plots as a shape-recognition backup for both LLM and human. Decimated
egress ships as `get_waveform`; rendered plots remain roadmap. The
principle is recorded so scalar-first reads as a scoped default, not the
end state.

**Scalar-guided zoom.** The waveforms this server sees — switching nodes,
amplifier internal nodes — are multi-scale: a piecewise-smooth backbone
plus sparse localized events (ringing spikes, coupling glitches,
crossover kinks). No single fixed-resolution view fits, so fidelity comes
from *navigation*, not from one lossy encoding. The overview is a
**stat-envelope**: per bucket, `[min, max, rms, mean]` — min/max
guarantees a narrow spike's amplitude survives bucketing, rms/mean
localize energy and drift. From the surfaced per-bucket facts the consumer
picks a sub-window and re-requests it finer — the same `get_waveform`
call with a narrower `[t_start, t_end]` — and recurses. Full-resolution
data stays on disk; detail enters context only when a measured quantity
earned the zoom. This is the engineer's own look-then-zoom loop, and it
reuses windowed egress plus the detectors (an envelope+carrier or
edge-metric descriptor is just the zoom-time payload for a flagged
window).

**Surface, don't judge — applied to navigation.** The stat-envelope and
any ranking obey the same rule as the rest of the result layer: the tool
surfaces *measured conditions* — crest factor, a bucket's spread relative
to its neighbors, `mean` drift, alternating-peak spacing, spectral
concentration — and may sort buckets by a named quantity, but it emits
**no phenomenon label and no trust verdict**. "Crest factor 8.1 here, 5.2×
the median bucket" is a fact; "spike / unstable / unreliable" is the
model's conclusion. Relative comparisons beat bare-magnitude thresholds
(one constant, one meaning). The model decides what the shape *is* and
where to look next; the server only makes the facts cheap to read.

**Optional always-attach plot.** A config default plus a per-call tool
parameter make waveform / analysis responses carry a rendered plot (MCP
`ImageContent`) of the queried window *alongside* the scalars and the
stat-envelope datapoints — not instead of them. The plot is the gestalt
layer for the vision branch above, and it is AI-zoomable by construction:
because it renders the *queried window*, the same narrowing zoom loop
re-renders it at finer scale. Off by default (token cost, non-vision
clients); on for clients that want shape-at-a-glance riding with every
result, with a per-call override either way.

### Decided egress & plot surface (2026-06-13)

The layered surface above resolves into three delivery channels, keyed by
*who consumes the output* — and the right channel for each is settled even
though one library choice is still open.

| Consumer | Needs | Channel |
|-|-|-|
| The agent (computes) | the numbers, full fidelity | `export_waveform` → CSV on disk |
| A vision model (one frame) | shape-at-a-glance | static PNG (`ImageContent`) attached to a result |
| A human (explores) | an interactive plot | `plot_waveform`, rendered to the richest surface the client supports |

A verified constraint shapes the last two: **the terminal CLIs do not render
images for the human.** Claude Code passes an `ImageContent` PNG to the
*model* (vision works) but never shows it to the user; Codex handles
tool-returned images unreliably (it tends to dump base64 into context). The
MCP "apps" widget surface (`ui://` HTML in a sandboxed iframe) is
**GUI-host-only** — Claude Desktop / claude.ai web, not the terminal. And
**interactivity benefits the human, not the LLM** — a model consumes a single
rendered frame, so zoom / pan / hover buys it nothing.

- **`export_waveform` (CSV) — full-fidelity egress to disk.** Returns a path,
  not data (full resolution in a response would blow context — the reason
  `get_waveform` decimates). Works on every analysis type. `.step` /
  Monte-Carlo runs are written **tidy / long** (`step_index, step_value, x,
  <signals…>`), which is forced rather than chosen: transient `.step` runs
  have a *different time vector per step*, so a wide shared-`x` layout is
  wrong. Complex AC traces default to **magnitude(dB) + phase(deg)** columns
  (lossless polar form, plot-ready, and the form the AC structural-analysis
  methods read), with `re/im` and `both` as a `complex_format` option. This
  is the *only* path that emits the full complex `H(f)` array — `get_waveform`
  rejects AC and `bode_metrics` returns scalars — so it is also the substrate
  the AC structural-analysis layer stands on. It stays a clean egress: **no**
  derived slope / group-delay / residual columns (those belong to the
  detector layer, not the substrate). Bounded by *windowing* (the
  scalar-guided-zoom loop), not silent decimation. No new dependencies.
  **Shipped.** Final column scheme: a unit-tagged x header (`time_s` /
  `freq_Hz` / `sweep`); real traces as their canonical name (`V(out)`);
  complex traces as `V(out)_mag_dB` + `V(out)_phase_deg` (or `_re`/`_im`, or
  all four). Phase is the **wrapped** `np.angle` (the lossless primitive — a
  consumer runs `np.unwrap` themselves). Non-finite samples are **kept** and
  counted (not dropped), so columns stay row-aligned. The CSV lands under
  `<circuit_dir>/.ltspice-mcp/waveforms/` (Linux-side, never beside a
  Windows-temp raw); a descending/non-monotonic axis is refused when windowed
  (searchsorted would corrupt it). Hardening from review: rows stream straight
  into the atomic temp file (no whole-CSV copy held in memory) under a generous
  row-count backstop that raises rather than truncates; the resolved output is
  rejected if a symlinked sidecar would redirect it outside the circuit
  directory; and in a stepped run a step whose axis misses the window is skipped
  with a surfaced fact rather than failing the whole export.
- **`plot_waveform` — adaptive, progressive enhancement.** One interactive
  chart core, two delivery wrappers chosen by the client detected at
  `initialize`: an in-chat **`ui://` widget** for an apps-capable GUI host
  (realistically Claude Desktop for a local stdio server), or render-to-HTML
  **opened on the local desktop** (probe an available opener — on WSL
  `explorer.exe` / `cmd.exe /c start` via a `wslpath -w` conversion, else
  `xdg-open` / `open` / `start`) for a terminal client — and *always* a text
  summary + the data path so the model
  keeps reasoning. Because the server is local, "interactive for co-design"
  needs no widget infrastructure: it just opens a window / browser the OS
  renders, which works identically under any CLI. The interactive chart is
  built on **uPlot** (~50 KB, zero-dependency, canvas-2D) inlined into one
  self-contained HTML — it inlines cleanly into both the offline file and the
  payload-capped widget and has the perf headroom to render full-fidelity
  transients; chosen over Plotly (~1.5 MB, heavy to inline per widget) because
  both inline size and tail perf favor it once the chart can receive full
  data. It is **not** a Python plotting dependency. **Fidelity:** full by
  default, a `max_points` parameter to override, and a min/max-preserving
  downsample that engages only above a high cap and is surfaced in
  `observations` (no silent truncation). **Shipped — both delivery tiers.**
  uPlot is vendored as a bundled MIT asset under `src/ltspice_mcp/assets/uplot/`
  (in the wheel, inlined at render — no pip dep, no CDN); a step whose axis misses
  the window is skipped, transient `.step` overlays with differing per-step time
  vectors are null-padded onto a union x, AC Bode phase is unwrapped, and
  non-finite samples become JSON `null` gaps. A global per-panel **cell cap**
  refuses (before allocating) a plot whose union-padded size would blow up.
  Delivery is chosen by the client detected at `initialize`
  (`capabilities.extensions["io.modelcontextprotocol/ui"]`):
  - **MCP Apps host (SEP-1865, Final 2026-01-26)** → an in-chat **`ui://`
    widget**. The canonical wiring, *not* inline embedding: the `plot_waveform`
    tool declares `_meta.ui.resourceUri`; one stable, predeclared renderer
    resource (`ui://ltspice-mcp/plot`, uPlot + the vendored Apache-2.0 ext-apps
    `App` runtime inlined) is served via `resources/read`; the host renders it in
    a sandboxed iframe and pipes a **compact** chart spec — decimated harder than
    the file, carried in the result **`_meta`** (a *non-model-visible* channel, so
    the plot still returns no numbers to the model) — which `app.ontoolresult`
    draws. The full-fidelity HTML still lands on disk; an oversized spec (byte cap)
    or a build failure falls back to local-open, surfaced as a fact.
  - **Terminal client** → the self-contained HTML is **opened locally**
    (`explorer.exe` via `wslpath -w`, else `xdg-open`/`open`/`startfile`, spawned
    detached so it never blocks).
  Both always return the file path + a text summary, so a host with neither
  surface degrades gracefully (the spec'd MCP Apps fallback).
- **Static PNG (the vision tier) — opt-in.** A config default
  `[analysis] attach_plot` (off) **plus** a per-call `attach_plot` tool
  parameter that overrides it: an operator can make a plot ride with every
  analysis result, while the model can opt in or out for a single call.
  Default off (a base64 PNG on every result is expensive, useless to
  non-vision clients, and barely works in Codex). Gated on the optional
  `[plot]` extra (matplotlib); absent → skip and note, don't error. Scoped to
  the tools where a plot adds gestalt first (`get_waveform`, `bode_metrics`).

**Fidelity by consumer.** `get_waveform` decimates (the LLM's context is the
limit), `export_waveform` is lossless (disk has no such limit), and
`plot_waveform` defaults to full fidelity (a browser is not context-bound),
capping only at the byte tail.

**Dependency line.** The core install stays lean. The interactive channel
adds **uPlot** (~50 KB, zero-dep, inlined) only (no Python plotting
dependency); **matplotlib is confined to the static-PNG tier and ships as an
optional `[plot]` extra.**
**Renderer principle:** the plot layer takes plain arrays + labels and is
ignorant of `job` / `RunRef` internals, so the three channels stay swappable
behind one data contract.

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
`annotations`, `profiles` (which tool profiles expose this tool), and an
optional `output_schema`. `SessionState.create()` filters tools by
profile during lifespan init; the dispatch table and tool definitions
both come from `registry`.

Key `lib/` modules:

|module|purpose|
|-|-|
|`services.py`|service layer shared by tools and resources — job resolution, cached result loading, reusable extraction|
|`sim_runner.py`, `sweep_runner.py`, `montecarlo_runner.py`|spicelib runner wrappers|
|`runner_manager.py`|centralized runner lifecycle with auto-invalidation on loop / simulator / output-folder change|
|`simulator.py`|simulator detection, WSL/Wine selection|
|`ltspice_wsl.py`, `wsl.py`|WSL path conversion and Windows interop|
|`raw_parser.py`, `log_parser.py`|simulation result parsing|
|`library_manager.py`, `library_parser.py`|component library handling|
|`symbol_geometry.py`|`.asy` symbol parsing, pin positions, rotation transforms, bounding boxes|
|`spice_lex.py`, `spice_lex_ops.py`, `spice_lex_views.py`|shared SPICE lexer pipeline — see [docs/spice_lex.md](spice_lex.md)|
|`pathutil.py`|path security (`safe_path()`, `resolve_safe_path()`)|

### Tool profiles

`config.tool_profile` controls which tools are exposed.

|profile|tool count|use case|
|-|-|-|
|`full` (default)|47|Claude Desktop, ChatGPT, web chat clients, non-agent LLMs, automation|
|`agentic`|37|Claude Code, Cursor, Windsurf, and other agents with native `Read`/`Edit`/`Write`|

The `agentic` profile drops 10 tools: the five netlist-editing wrappers
(`create_netlist`, `read_circuit`, `set_component_value`, `parameter`,
`edit_directive`) — things a capable agent does natively via filesystem
access — the three library session tools (`load_library`,
`unload_library`, `list_libraries`), and the `configure_sweep` /
`configure_montecarlo` config builders. It keeps simulation lifecycle,
binary `.raw` parsing and analysis, batch run/results, library search
(`find_model`), and the schematic toolset an agent cannot replicate
by editing text — geometry-aware editing with orthogonal routing and
pin-collision/junction checks: `create_schematic`, `add_component`,
`apply_schematic_ops`, `connect`, `export_netlist`,
`reset_schematic`, `symbol_info`, `component_info`, `trace_net`. The
ack-only mutations (`move_component`, `remove_component`,
`set_component_attribute`, `add_net_label`, `remove_net_label`,
`remove_wire`) are `apply_schematic_ops` ops, not standalone tools.

Set via `[tools] profile` in `ltspice-mcp.toml` or the
`LTSPICE_MCP_TOOL_PROFILE` env var. Error hints adapt to the active
profile so they never point at unavailable tools.

## Backend: spicelib

`spicelib` (>= 1.4.9) is the core Python library for SPICE automation,
by Nuno Brum. PyLTSpice is a thin re-export wrapper over spicelib that
adds nothing — we depend on `spicelib` directly.

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

- **LTspice** supports `.asc` → `.net` conversion via `create_netlist()`.
  macOS LTspice has no CLI switch support. Default switches: `-Run -b`.
- **NGspice** has a compatibility mode (`kiltpsa` default for
  KiCad/LTspice/PSPICE). Default switches: `-b -o -r -a`. Native on Linux.
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
  extension (`.cir`, `.net`, `.sp`). `sim_runner.py` preserves the
  original extension in `run_filename`.

`simulator.path` in `ltspice-mcp.toml` must point at the
Windows-side executable (e.g.
`/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) — spicelib can't
auto-detect across the WSL boundary.

## Diagnostics

`log_parser.py:extract_log_diagnostics()` extracts structured warnings
and errors from simulator log files — parse errors with caret pointers,
fatal errors, convergence messages, `.MEAS` parse failures — and
`run_simulation`, `check_job`, and `simulation_summary` attach them to
their responses as `warnings` / `errors` lists. An agent driving raw
`ngspice -b` gets a 200-line log dump and has to grep; here the same
failure arrives inside the tool response.

Above the raw lists sits the observation surfacer
(`lib/result_observations.py`), which lifts salient facts into a
curated `observations` list on the run summary:

|observation kind|what it surfaces|
|-|-|
|`relay`|the simulator's own error lines, with the simulator's severity (never an invented one)|
|`reconciliation`|requested `.meas`/`.four` outputs that were not produced|
|`value`|non-finite samples and extreme node values in the trace data|
|`coverage`|checks that were skipped, so a thin result can't pass as a clean one|

The surfacer deliberately renders no trust verdict — it surfaces facts
for the model to judge. An empty `observations` list means nothing
tripped a check, not that the result is verified correct.

**Scope:** LTspice and ngspice both get this first-class — ngspice's
stdout diagnostics are captured alongside its log file and folded into
the same pipeline. qspice / xyce get best-effort convergence and
fatal-error parsing.

## Non-goals

These are intentional gaps, not pending features. Adopt accordingly.

- **Auto-routing / auto-placement.** Conflict-set returns on refusal
  are the substitute. Routing is a research project, not a tool.
- **Semantic part search** ("find me a low-Vgs-th NMOS under 100mΩ
  Rds(on)"). Requires parsing every `.model` card, normalizing units
  across vendors, building a queryable parameter index — a separate
  parameter-database effort.
- **Thin wrappers around file-edit operations.** The `agentic` profile
  already drops these for clients with native `Read`/`Edit`.
- **Per-simulator parity for geometry.** LTspice and ngspice are
  co-equal for simulation, result parsing, diagnostics, and analysis.
  The geometry layer (`.asc`, `.asy`, symbol coordinates) stays
  LTspice-only by nature — ngspice has no schematic format to edit.
  QSPICE and Xyce get best-effort support through spicelib's common
  interface.
- **Auth / multi-tenancy / remote-MCP plumbing.** Local-tool territory.
  No OAuth flows, no per-tenant isolation.
- **Generic cross-simulator EDA features** that PyLTSpice and ngspice's
  CLI already cover well.

## Roadmap

Forward-looking and **not commitments**. Direction reflects the same
spec at the top of this doc — investment compounds where structured +
validated + geometry-aware + LTspice-specific overlap.

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
- **Waveform egress & plotting**: the surface is now specified (see *Decided
  egress & plot surface* above) — `export_waveform` (full-fidelity CSV, all
  analysis types) first, then the adaptive `plot_waveform` (terminal
  local-open before the GUI-host `ui://` widget), then the opt-in static-PNG
  attach for the vision tier. What remains is implementation (uPlot for the
  interactive tier). Resolves the remaining half of the scalar-only rigidity
  noted under *Waveforms* above.
- **Pin-compatible alternate suggestions** for unknown parts.
- **`schematic` profile** (geometry tools only, no simulation) for
  layout-focused agents — pending a real user request, since each
  profile is a test-matrix axis.

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
3. Run AC simulation
4. `bode_metrics(mode="filter")` — verify the -3dB point near 1.59kHz
5. Change R to 10k (fc → ~159Hz)
6. Re-simulate, re-run `bode_metrics` — verify the bandwidth shifted
7. Load a custom library, use a component from it in a new circuit

If any step fails, the doctor tool (see Roadmap) is the eventual answer;
until it ships, check `simulator.path` in TOML and confirm
`ltspice-mcp.toml`'s `[security] allowed_paths` includes your working
directory.
