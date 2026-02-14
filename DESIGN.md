# ltspice-mcp — Design Document

## 1. Overview

An MCP server for LTSpice circuit simulation, with support for other SPICE
simulators (NGspice, QSPICE, Xyce) where available. Gives LLMs the ability to
edit netlists, run simulations, extract waveform data, take measurements, and
generate plots.

Built on the official `mcp` Python SDK (low-level `Server` API) with `spicelib`
as the simulation backend.

## 2. Why MCP vs. asking the LLM to use spicelib directly?

An LLM with code execution (Claude Code, Cursor) could write spicelib Python
scripts directly. MCP adds value in specific ways:

- **Works without code execution**: Claude Desktop, ChatGPT, web chat interfaces
  have no shell or Python. MCP tools are the only way to give them simulation
  capabilities. This is the primary audience.
- **Reliability**: `run_simulation(netlist="foo.cir")` is a tested code path.
  LLM-generated spicelib code will have import mistakes, wrong method names,
  forgotten `save_netlist()` calls, etc.
- **Context efficiency**: a tool call is ~50 tokens. Equivalent Python is
  20-40 lines. Over an iterative design session (edit → simulate → analyze →
  plot → repeat), this compounds.
- **Images inline**: MCP returns base64 PNG in the protocol. Bode plots and
  waveforms appear in the conversation. With raw Python the LLM saves to file
  but can't display it in most clients.
- **No user setup**: user installs the MCP server once. No need for spicelib
  in their project venv, no Python knowledge required.

Where MCP is weaker: fixed tool set (less flexible than arbitrary Python),
extra process to maintain.

## 3. Goals

- **Complete feedback loop**: design a circuit, simulate it, read back actual
  voltages/currents/frequencies, iterate on the design.
- **LTSpice-first**: LTSpice is the primary target. Other simulators (NGspice,
  QSPICE, Xyce) are supported through spicelib's common interface but
  LTSpice-specific features (`.asc` schematics, component library, log
  parsing) are first-class.
- **Other simulators where useful**: NGspice runs natively on Linux
  (`apt install ngspice`) — no Wine needed. Useful fallback.
- **Structured data**: typed, machine-readable results — not pass/fail strings.
- **Visualization**: waveform plots and Bode diagrams as inline images.
- **Production quality**: timeouts, progress reporting, actionable errors.

## 4. Backend: spicelib

`spicelib` (>= 1.4.7) is the core Python library for SPICE simulator
automation, by Nuno Brum. (PyLTSpice is a thin re-export wrapper over spicelib
that adds nothing — we depend on `spicelib` directly.)

### Simulator classes

All four share the same base `Simulator` ABC with a uniform interface:

| Simulator | Class | Import | Platform |
|-----------|-------|--------|----------|
| LTSpice | `LTspice` | `spicelib.simulators.ltspice_simulator` | Windows native, Linux/macOS via Wine |
| NGspice | `NGspiceSimulator` | `spicelib.simulators.ngspice_simulator` | Linux/macOS/Windows native |
| QSPICE | `Qspice` | `spicelib.simulators.qspice_simulator` | Windows only (Wine support limited) |
| Xyce | `XyceSimulator` | `spicelib.simulators.xyce_simulator` | Linux/Windows native |

The `Simulator` base class (in `spicelib.sim.simulator`) defines:

```python
class Simulator(ABC):
    spice_exe: list[str]       # executable path (may include 'wine' prefix)
    process_name: str
    raw_extension: str         # '.raw' (or '.qraw' for QSPICE)

    @classmethod
    def run(cls, netlist_file, cmd_line_switches, timeout, ...) -> int
    @classmethod
    def valid_switch(cls, switch, parameter) -> list
    @classmethod
    def is_available(cls) -> bool
    @classmethod
    def create_from(cls, path_to_exe, process_name) -> Simulator
    @classmethod
    def get_default_library_paths(cls) -> list[str]
```

Each simulator auto-detects its executable at import time by scanning known
installation paths. All produce `.raw` + `.log` output (QSPICE uses `.qraw`).

### Core components

| Component | Import | Purpose |
|-----------|--------|---------|
| `SpiceEditor` | `spicelib.editor.spice_editor` | Read/modify/write `.net`/`.cir` netlists |
| `AscEditor` | `spicelib.editor.asc_editor` | Read/modify `.asc` schematics (LTSpice) |
| `SimRunner` | `spicelib.sim.sim_runner` | Batch execution with parallel sim support |
| `RawRead` | `spicelib.raw.raw_read` | Parse binary `.raw`/`.qraw` waveform output |
| `RawWrite` | `spicelib.raw.raw_write` | Create synthetic `.raw` files |
| `LTSpiceLogReader` | `spicelib.log.ltsteps` | Extract `.MEAS`, step data, Fourier |
| `SimStepper` | `spicelib.sim.sim_stepping` | Multi-dimensional parameter sweeps |
| `Montecarlo` | `spicelib.sim.tookit.montecarlo` | Statistical tolerance analysis |
| `WorstCaseAnalysis` | `spicelib.sim.tookit.worst_case` | Exhaustive min/max enumeration |

### Key characteristics

- **File-based interaction**: all simulator communication is through files
  (`.net`/`.cir` in, `.raw` + `.log` out). No GUI automation.
- **RawRead dialect auto-detection**: `RawRead(file, dialect=None)` auto-detects
  the simulator from the file. Supports `"ltspice"`, `"ngspice"`, `"qspice"`,
  `"xyce"` dialects explicitly if needed.
- **Engineering notation**: handles SI prefixes natively (`1k`, `100n`, `3.3Meg`).
- **Hierarchical access**: subcircuit components via colon paths (`XU1:R1`).
- **Lazy RAW loading**: traces are only parsed when accessed.

### Simulator-specific notes

- **LTSpice**: supports `.asc` → `.net` conversion via `create_netlist()`.
  Native macOS LTSpice has no CLI switch support. Default run switches:
  `-Run -b`.
- **NGspice**: has a compatibility mode setting (`kiltpsa` default for
  KiCad/LTSpice/PSPICE compatibility). Default run switches: `-b -o -r -a`.
  Natively available on Linux.
- **QSPICE**: uses `.qraw` extension (double precision). Windows-only, limited
  Wine support. Default run switch: `-o`.
- **Xyce**: Sandia's parallel SPICE. Supports `-syntax` and `-norun` for
  validation without simulation. Default run switches: `-l -r`.

## 5. MCP SDK: Low-Level Server API

We use `mcp.server.lowlevel.Server` (not FastMCP) for full control over:
- Tool registration and JSON schema definition
- Result serialization (mixed text + image responses)
- Long-running simulation lifecycle
- Error response formatting

### Why not FastMCP?

FastMCP auto-generates schemas from type annotations. For this project:
- We need explicit control over how waveform data and images are serialized
- Simulation job management requires custom patterns
- The convenience savings are marginal given the complexity
- Fewer abstraction layers, fewer surprises

### Server skeleton

Uses the MCP SDK's `lifespan` API for state lifecycle and a module-based
dispatch table instead of a monolithic match statement.

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mcp.server.stdio
from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.tools import circuit, simulation, analysis, visualization, advanced, library

# Each tool module exports TOOL_DEFS and TOOL_HANDLERS
ALL_MODULES = [circuit, simulation, analysis, visualization, advanced, library]

# Build unified dispatch table at import time
_DISPATCH: dict[str, Any] = {}
for _mod in ALL_MODULES:
    _DISPATCH.update(_mod.TOOL_HANDLERS)

@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Initialize session state on startup, clean up on shutdown."""
    config = ServerConfig.from_env()
    available = detect_simulators()
    state = SessionState.create(config, available)
    try:
        yield {"state": state}
    finally:
        state.shutdown()

server = Server("ltspice-mcp", lifespan=server_lifespan)

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [tool for mod in ALL_MODULES for tool in mod.TOOL_DEFS]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.Content]:
    handler = _DISPATCH.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    state = server.request_context.lifespan_context["state"]
    return await handler(arguments, state)
```

### Tool module convention

Each `tools/*.py` module exports two names:

```python
# tools/circuit.py
from ltspice_mcp.state import SessionState
import mcp.types as types

async def handle_create_netlist(args: dict, state: SessionState) -> list[types.Content]:
    ...

async def handle_read_netlist(args: dict, state: SessionState) -> list[types.Content]:
    ...

TOOL_DEFS: list[types.Tool] = [
    types.Tool(name="create_netlist", description="...", inputSchema={...}),
    types.Tool(name="read_netlist", description="...", inputSchema={...}),
]

TOOL_HANDLERS: dict[str, Any] = {
    "create_netlist": handle_create_netlist,
    "read_netlist": handle_read_netlist,
}
```

## 6. Tools

### 6.1 Circuit Management

#### `create_netlist`
Create a new SPICE netlist file from a content string.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | yes | Project/file name (without extension) |
| `content` | string | yes | Complete SPICE netlist content |

Returns: file path of created `.cir` file, component count summary.

#### `read_netlist`
Read and return the contents of an existing netlist.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Path to `.net`/`.cir` file |

Returns: netlist text content, parsed component list.

#### `list_components`
List all components in a netlist, optionally filtered by type.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `prefix` | string | no | Filter by prefix (R, C, L, Q, M, X, V, I, etc.) |

Returns: array of `{reference, value, type}`.

Uses: `SpiceEditor.get_components()`

#### `get_component_value`
Get the current value of a specific component.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `reference` | string | yes | Component reference (e.g., `"R1"`, `"XU1:C2"`) |

Returns: component value string (engineering notation).

Uses: `SpiceEditor.get_component_value()`

#### `set_component_value`
Modify a component's value in the netlist.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `reference` | string | yes | Component reference designator |
| `value` | string | yes | New value (e.g., `"10k"`, `"100n"`, `"LM358"`) |

Returns: confirmation with old and new values.

Uses: `SpiceEditor.set_component_value()`

#### `set_component_values`
Batch-modify multiple component values.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `values` | object | yes | Map of reference → value (e.g., `{"R1": "10k", "C1": "100n"}`) |

Returns: summary of all changes.

Uses: `SpiceEditor.set_component_values()`

#### `set_parameter`
Set a `.PARAM` directive value.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `name` | string | yes | Parameter name |
| `value` | string | yes | Parameter value |

Uses: `SpiceEditor.set_parameter()`

#### `get_parameters`
List all `.PARAM` names and their current values.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |

Returns: `{name: value}` map.

Uses: `SpiceEditor.get_all_parameter_names()` + `get_parameter()`

#### `add_instruction`
Add a SPICE directive to the netlist. If a unique directive of the same type
exists (e.g., `.tran`), it is replaced automatically by spicelib.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `instruction` | string | yes | SPICE directive (e.g., `.tran 0 10m 0 1u`) |

Uses: `SpiceEditor.add_instruction()`

#### `remove_instruction`
Remove a SPICE directive from the netlist.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `instruction` | string | yes | Directive text or regex pattern |

Uses: `SpiceEditor.remove_instruction()` / `remove_Xinstruction()`

#### `convert_schematic`
Convert an `.asc` schematic file to a `.net` netlist. LTSpice-only (requires
the LTSpice executable for `.asc` → `.net` conversion).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `asc_path` | string | yes | Path to `.asc` file |

Returns: path to generated `.net` file.

Uses: `LTspice.create_netlist()` (via `SimRunner`)

### 6.2 Simulation Execution

#### `run_simulation`
Run a simulation synchronously with timeout and progress reporting.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `timeout` | number | no | Timeout in seconds (default: 300) |
| `simulator` | string | no | Override simulator (`"ltspice"`, `"ngspice"`, `"qspice"`, `"xyce"`) |
| `switches` | array[string] | no | Simulator CLI switches |

Returns: `{status, raw_file, log_file, duration_s, simulator_used, warnings[]}`.

Uses: `SimRunner.run_now()`

Error handling:
- Timeout → suggestion to reduce sim time or simplify circuit
- Non-convergence → parse log for "Time step too small"
- Singular matrix → suggest checking for floating nodes
- Missing subcircuit → report which model is missing

#### `run_simulation_async`
Start a simulation in the background, return immediately with a job ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `timeout` | number | no | Timeout in seconds (default: 600) |
| `simulator` | string | no | Override simulator |

Returns: `{job_id, status: "running"}`.

Uses: `SimRunner.run()` (non-blocking with callback)

#### `get_simulation_status`
Check the status of an async simulation job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | yes | Job ID from `run_simulation_async` |

Returns: `{job_id, status, raw_file, log_file, error}`.

#### `list_simulations`
List all simulation jobs (active and completed) in the current session.

Returns: array of `{job_id, netlist, status, started_at, completed_at}`.

### 6.3 Result Analysis

#### `get_trace_names`
List all signal/trace names available in a simulation result.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw`/`.qraw` file |

Returns: `{plot_type, traces: [{name, type}], n_points, n_steps, dialect}`.

Uses: `RawRead.get_trace_names()`, `get_plot_name()`, `.dialect`

#### `get_waveform_data`
Extract numeric waveform data for one or more signals.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw`/`.qraw` file |
| `traces` | array[string] | yes | Signal names (e.g., `["V(out)", "I(R1)"]`) |
| `step` | integer | no | Step index for stepped simulations (default: 0) |
| `downsample` | integer | no | Return every Nth point (for large datasets) |
| `time_range` | array[number] | no | `[start, end]` time window filter |

Returns: `{axis: {name, unit, data[]}, traces: [{name, unit, data[]}]}`.

For AC analysis, trace data contains `{magnitude[], phase_deg[]}` instead
of raw complex values.

Auto-downsamples to 10k points if trace exceeds that, with a note in the
response. The `downsample` parameter overrides this.

Uses: `RawRead.get_trace()`, `get_wave()`, `get_axis()`

#### `get_measurements`
Extract `.MEAS` directive results from the simulation log.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `log_file` | string | yes | Path to `.log` file |

Returns: `{measurements: [{name, value, unit}], step_count}`.

For stepped simulations, returns measurements indexed by step.

Uses: `LTSpiceLogReader`

#### `get_operating_point`
Read DC operating point data (node voltages, branch currents).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.op.raw` file |

Returns: `{nodes: {name: voltage}, currents: {name: current}}`.

#### `get_node_voltage`
Get the voltage at a specific node at a specific time or frequency.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw` file |
| `node` | string | yes | Node name (e.g., `"out"`, `"V(out)"`) |
| `at` | number | no | Time or frequency point (numpy interpolation) |
| `step` | integer | no | Step index |

Returns: `{node, time_or_freq, value, unit}`.

#### `get_fourier_data`
Extract Fourier analysis results from the log.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `log_file` | string | yes | Path to `.log` file |
| `signal` | string | no | Specific signal (default: all) |

Returns: array of `{harmonic, frequency, magnitude, phase, thd}`.

Uses: `LTSpiceLogReader.fourier`

#### `get_simulation_summary`
High-level summary of a completed simulation.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw` file |
| `log_file` | string | no | Path to `.log` file (auto-detected if omitted) |

Returns: `{sim_type, dialect, n_nodes, n_points, n_steps, measurements[], warnings[]}`.

### 6.4 Visualization

All visualization tools return `ImageContent` (base64 PNG) plus a
`TextContent` with a textual summary, so the LLM can reason about results
even without seeing the image.

#### `plot_waveform`
Plot one or more signals vs time (transient) or frequency (AC).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw` file |
| `traces` | array[string] | yes | Signal names to plot |
| `step` | integer | no | Step index |
| `time_range` | array[number] | no | `[start, end]` X-axis limits |
| `title` | string | no | Plot title |

Returns: PNG + text summary (min/max/mean of each trace).

#### `plot_bode`
Bode plot (magnitude + phase) from AC analysis.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw` file (AC analysis) |
| `trace` | string | yes | Signal name |
| `step` | integer | no | Step index |

Returns: PNG (dual subplot: dB vs freq, degrees vs freq) + text summary
(DC gain, -3dB bandwidth, phase margin, gain margin).

#### `plot_histogram`
Histogram of a measurement across Monte Carlo or sweep runs.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `log_file` | string | yes | Path to `.log` file |
| `measurement` | string | yes | Measurement name |
| `bins` | integer | no | Number of bins (default: 50) |

Returns: PNG + text summary (mean, std, min, max, sigma levels).

#### `plot_xy`
Generic X-Y plot of any two traces.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_file` | string | yes | Path to `.raw` file |
| `x_trace` | string | yes | X-axis signal |
| `y_trace` | string | yes | Y-axis signal |
| `step` | integer | no | Step index |

Returns: PNG.

### 6.5 Advanced Analysis

#### `setup_parameter_sweep`
Configure a multi-parameter sweep. Uses spicelib's `SimStepper` which
overcomes the 3-parameter `.STEP` limit of LTSpice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `sweeps` | array[object] | yes | `[{param, start, stop, step}]` or `[{param, values: [...]}]` |

Returns: `{total_simulations, sweep_config}`.

Uses: `SimStepper.add_param_sweep()` / `add_value_sweep()`

#### `setup_monte_carlo`
Configure Monte Carlo analysis with component tolerances.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to netlist file |
| `tolerances` | object | yes | `{"R": 0.01, "C": 0.05}` or `{"R1": 0.05}` |
| `distribution` | string | no | `"uniform"` or `"normal"` (default: `"uniform"`) |
| `num_runs` | integer | yes | Number of iterations |

Returns: `{components_affected, total_runs, distribution}`.

Uses: `Montecarlo.set_tolerance()`

#### `run_batch_analysis`
Execute a previously configured sweep or Monte Carlo analysis.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `netlist` | string | yes | Path to configured netlist |
| `analysis_type` | string | yes | `"sweep"` or `"monte_carlo"` |
| `timeout` | number | no | Total timeout in seconds |

Returns: `{completed_runs, failed_runs, measurements_summary}`.

#### `get_sweep_results`
Query sweep/MC results filtered by parameter values.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `log_file` | string | yes | Path to log file |
| `measurement` | string | yes | Measurement name |
| `filters` | object | no | `{"R1": "1k"}` — filter by parameter value |

Returns: `{values: [{params, measurement_value}], stats: {min, max, mean, std}}`.

### 6.6 Component Library

#### `search_components`
Search the simulator's built-in component/model library **and** any loaded
libraries (see `load_library` below).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Search term (e.g., `"LM358"`, `"NMOS"`, `"schottky"`) |
| `category` | string | no | Filter: `"opamp"`, `"bjt"`, `"mosfet"`, `"diode"`, `"jfet"`, `"subcircuit"` |
| `max_results` | integer | no | Limit (default: 20) |

Returns: array of `{name, category, file_path, loaded}`.

The `loaded` flag indicates whether the component comes from a loaded
library (ready to use) vs. the simulator's built-in paths.

Uses: `Simulator.get_default_library_paths()` and
`search_file_in_containers()` for built-in search,
`SessionState.libraries` for loaded libraries.

#### `get_model_info`
Get SPICE model parameters for a component.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | yes | Model name (e.g., `"2N2222"`) |

Returns: `{name, type, parameters: {}, source_file}`.

#### `list_subcircuits`
List available subcircuit models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | no | Filter by subdirectory/category |

Returns: array of `{name, file, pin_count}`.

### 6.7 Library Management

SPICE model and subcircuit definitions (`.MODEL`, `.SUBCKT`) can live in any
text file — `.lib`, `.mod`, `.sub`, `.txt`, `.inc`, `.cir`, or no extension.
These tools parse the file content for SPICE directives regardless of
extension.

#### `load_library`
Parse a file containing SPICE model/subcircuit definitions, index its
contents, and keep it in the session. Returns a compact summary of
everything found so the LLM knows what components are available.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Path to any text file containing `.MODEL`/`.SUBCKT` definitions |

Returns: `{path, models: [{name, type}], subcircuits: [{name, pins}], total_components}`.

The summary is small enough to stay in conversation context. Full
definitions are stored in session state for on-demand lookup via
`get_library_component`.

#### `get_library_component`
Get the full SPICE definition of a specific component from any loaded
library. Use this to inspect pin order, default parameters, or the raw
model text before wiring up a component.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | yes | Model or subcircuit name (e.g., `"LM358"`, `"2N2222"`) |

Returns: `{name, type, definition, pins, source_file, include_directive}`.

The `include_directive` field (e.g., `.include /path/to/opamp.lib`) is
ready to paste into a netlist — this is the key ergonomic win.

#### `list_loaded_libraries`
List all libraries currently loaded in the session.

Returns: array of `{path, model_count, subcircuit_count}`.

#### `unload_library`
Remove a library from the session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Path of the library to unload |

#### Typical LLM workflow

```
1. User: "Design an amp using the TI opamp library"
2. LLM calls: load_library(path="ti_opamps.lib")
   → {models: [], subcircuits: [{name: "OPA2277", pins: ["+","-","V+","V-","OUT"]}, ...]}
   → LLM now knows what's available and pin ordering
3. LLM calls: get_library_component(name="OPA2277")
   → {definition: ".SUBCKT OPA2277 ...", pins: [...], include_directive: ".include /circuits/ti_opamps.lib"}
4. LLM calls: create_netlist(content="...\n.include /circuits/ti_opamps.lib\n...")
   → Wires up the component correctly using the pin order from step 3
```

## 7. MCP Resources

| URI Pattern | Description | MIME Type |
|---|---|---|
| `ltspice://netlists` | List all netlists in the working directory | application/json |
| `ltspice://netlists/{name}` | Read a netlist file's contents | text/plain |
| `ltspice://results/{name}/summary` | Simulation result summary | application/json |
| `ltspice://results/{name}/log` | Raw log file contents | text/plain |
| `ltspice://library/models` | Available built-in models | application/json |
| `ltspice://config` | Current server configuration | application/json |

## 8. MCP Prompts

#### `design_filter`
Arguments: `filter_type` (lowpass/highpass/bandpass/notch), `cutoff_freq`,
`order`, `topology` (optional).

Guides: component calculation → netlist creation → AC simulation → Bode plot
→ iterate if specs not met.

#### `analyze_amplifier`
Arguments: `netlist_path`, `input_node`, `output_node`.

Guides: AC analysis for gain/bandwidth → transient for slew rate → DC
operating point for bias.

#### `tolerance_analysis`
Arguments: `netlist_path`, `measurement_name`, `target_value`,
`acceptable_range`.

Guides: Monte Carlo setup → batch run → histogram → yield percentage.

#### `debug_simulation`
Arguments: `netlist_path`, `error_description`.

Guides: syntax check → connectivity check → simplified run → log analysis.

## 9. Architecture

```
ltspice-mcp/
  pyproject.toml                 # uv-managed, hatchling build
  src/
    ltspice_mcp/
      __init__.py
      __main__.py                # python -m ltspice_mcp support
      server.py                  # Server instance, lifespan, tool/resource/prompt assembly
      main.py                    # Entry point: parse args, select transport (stdio/sse), run
      config.py                  # ServerConfig from env vars
      state.py                   # SessionState dataclass + SimulationJob
      errors.py                  # Structured error hierarchy

      tools/
        __init__.py              # collect_tools() helper that merges all modules
        _base.py                 # Shared helpers: path validation, to_thread wrappers
        circuit.py               # 10 tools: netlist CRUD, components, parameters, directives
        simulation.py            # 4 tools: run, async run, status, list
        analysis.py              # 7 tools: traces, waveforms, measurements, OP, Fourier
        visualization.py         # 4 tools: waveform, bode, histogram, xy
        advanced.py              # 4 tools: sweeps, monte carlo, batch
        library.py               # 7 tools: search, model info, subcircuits, load/unload/list/get library

      lib/
        simulator.py             # detect_simulators(), SimRunner wrapper
        cache.py                 # Generic FileCache[T] with mtime invalidation
        plotting.py              # Matplotlib rendering → PNG bytes (sync, pure functions)
        format.py                # Engineering notation, unit formatting
        pathutil.py              # resolve_safe_path() — sandboxed path resolution
        library_parser.py        # Parse .MODEL/.SUBCKT from any text file

      resources.py               # All MCP resource handlers
      prompts.py                 # All MCP prompt templates

  tests/
    conftest.py                  # Fixtures: sample netlists, mock .raw/.log files
    test_circuit.py
    test_simulation.py
    test_analysis.py
    test_visualization.py
    test_library.py
    test_cache.py
    test_pathutil.py
```

### Async Bridging

spicelib is entirely synchronous — file I/O, subprocess calls, numpy array
operations. The MCP server runs on asyncio. Without bridging, any blocking
call freezes the server (no other tool calls can be served, heartbeats stop).

**Rule**: every spicelib call that touches the filesystem or waits on a
process must go through `asyncio.to_thread()`.

| Operation | Blocking? | Bridge strategy |
|---|---|---|
| `SpiceEditor.set_component_value()` | No (in-memory) | Call directly |
| `SpiceEditor.save_netlist()` | Yes (file write) | `asyncio.to_thread()` |
| `SpiceEditor()` constructor | Yes (file read + parse) | `asyncio.to_thread()` |
| `SimRunner.run_now()` | Yes (subprocess + join) | `asyncio.to_thread()` |
| `SimRunner.run()` | No (spawns thread) | Call directly, but see callback bridging below |
| `RawRead()` with traces | Yes (binary file I/O) | `asyncio.to_thread()` |
| `LTSpiceLogReader()` | Yes (file parse) | `asyncio.to_thread()` |
| `Simulator.is_available()` | Negligible | Call directly at startup |

**Callback bridging for async simulations**: `SimRunner.run()` spawns a
worker thread and fires its callback *in that worker thread*, not the asyncio
event loop. To bridge:

```python
import asyncio

def _on_sim_complete(raw_path, log_path, job: SimulationJob):
    """Called in SimRunner's worker thread."""
    job.raw_file = raw_path
    job.log_file = log_path
    job.status = "completed"
    job.completed_at = datetime.now()
    # Wake up any asyncio waiters (thread-safe)
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(job.done_event.set)
```

In the tool handler:

```python
async def handle_get_simulation_status(args, state):
    job = state.jobs[args["job_id"]]
    if job.status == "running":
        # Optionally wait briefly for completion
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(job.done_event.wait(), timeout=0.1)
    return [types.TextContent(type="text", text=format_job_status(job))]
```

### Path Security

Tool arguments include file paths (`netlist`, `raw_file`, `log_file`).
Without validation, an LLM could request `read_netlist(path="/etc/shadow")`.

All file-accepting tools must validate paths via `pathutil.resolve_safe_path()`:

```python
# lib/pathutil.py
from pathlib import Path
from ltspice_mcp.errors import PathSecurityError

def resolve_safe_path(user_path: str, working_dir: Path) -> Path:
    """Resolve a user-supplied path, ensuring it stays within working_dir.

    - Resolves relative paths against working_dir
    - Resolves symlinks and '..' components
    - Raises PathSecurityError if the resolved path escapes working_dir
    """
    candidate = (working_dir / user_path).resolve()
    if not candidate.is_relative_to(working_dir.resolve()):
        raise PathSecurityError(
            f"Path '{user_path}' resolves outside working directory"
        )
    return candidate
```

This is enforced in `tools/_base.py` so individual tool handlers never see
raw user paths.

### Simulator Resolution

At startup, probe all four simulators for availability:

```python
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

SIMULATORS = {
    "ltspice": LTspice,
    "ngspice": NGspiceSimulator,
    "qspice": Qspice,
    "xyce": XyceSimulator,
}

def detect_simulators() -> dict[str, type[Simulator]]:
    return {name: cls for name, cls in SIMULATORS.items() if cls.is_available()}
```

The config selects a default simulator (env var `LTSPICE_SIMULATOR` or first
available). Individual `run_simulation` calls can override with the
`simulator` parameter.

### Session State

```python
@dataclass
class SessionState:
    config: ServerConfig
    available_simulators: dict[str, type[Simulator]]
    default_simulator: type[Simulator]
    runner: SimRunner                        # shared, supports parallel sims
    editors: FileCache[SpiceEditor]          # path → cached editor (mtime-checked)
    results: FileCache[RawRead]             # path → cached parsed results (mtime-checked)
    jobs: dict[str, SimulationJob]          # job_id → async sim tracking
    libraries: dict[Path, LoadedLibrary]    # path → parsed library index
    working_dir: Path

    @classmethod
    def create(cls, config: ServerConfig, available: dict) -> "SessionState":
        """Factory used by server lifespan."""
        ...

    def shutdown(self) -> None:
        """Cancel pending jobs, clean up runner."""
        ...
```

### Generic File Cache

`editor_pool` and `result_cache` from the original design share the same
pattern: a `dict[Path, T]` with mtime-based invalidation. A single generic
class handles both.

```python
# lib/cache.py
from pathlib import Path
from typing import Generic, TypeVar, Callable

T = TypeVar("T")

class FileCache(Generic[T]):
    """Cache keyed by file path, invalidated when file mtime changes."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[float, T]] = {}  # path → (mtime, value)

    def get(self, path: Path, factory: Callable[[Path], T]) -> T:
        """Return cached value, or create via factory if stale/missing."""
        mtime = path.stat().st_mtime
        entry = self._entries.get(path)
        if entry is not None and entry[0] == mtime:
            return entry[1]
        value = factory(path)
        self._entries[path] = (mtime, value)
        return value

    def invalidate(self, path: Path) -> None:
        self._entries.pop(path, None)
```

Usage:
- `state.editors.get(path, lambda p: SpiceEditor(str(p)))` — cached editor
- `state.results.get(path, lambda p: RawRead(str(p)))` — cached raw reader

### Loaded Library Index

`load_library` parses any text file for `.MODEL` and `.SUBCKT` directives
regardless of file extension. The parsed data is stored in session state
for fast lookup.

#### What spicelib provides (use, don't reimplement)

spicelib has significant library infrastructure we leverage:

- **`SpiceCircuit.find_subckt_in_lib(library, subckt_name)`** — parses a
  library file and extracts a specific `.SUBCKT` definition by name. Returns
  a `SpiceCircuit` object. Used by `get_library_component` for subcircuit
  detail lookups.
- **`SpiceCircuit.find_subckt_in_included_libs(name)`** — follows `.include`
  and `.lib` directives in a netlist, resolves file paths, and finds
  subcircuits transitively. Used internally when netlists reference external
  models.
- **`Simulator.get_default_library_paths()`** — returns the simulator's
  built-in library search paths (handles Wine translation on Linux/macOS).
- **`search_file_in_containers(filename, *containers)`** — resolves library
  file paths including inside `.zip` archives.
- **`BaseEditor.set_custom_library_paths(*paths)`** — registers additional
  library search paths (class-level, affects all instances).
- **`get_subcircuit_names()`** / **`get_subcircuit_named(name)`** — query
  subcircuits defined within a loaded netlist/file.

#### What we build ourselves

spicelib has **no `.MODEL` parsing** (there are TODO comments in the source
about this) and **no library enumeration** — it can find a specific
subcircuit by name but cannot list everything in a file. We need:

1. **`.MODEL` extraction** — regex-based, since spicelib has none
2. **Library enumeration** — "what's in this file?" to build the summary
   returned by `load_library`
3. **Indexing layer** — session-level dict mapping component names to their
   definitions across all loaded libraries

```python
# lib/library_parser.py
import re
from dataclasses import dataclass, field
from pathlib import Path
from spicelib.editor.spice_editor import SpiceCircuit

@dataclass
class ModelDefinition:
    name: str
    type: str              # NPN, PNP, NMOS, PMOS, D, etc.
    definition: str        # raw .MODEL line(s)
    source_file: Path

@dataclass
class SubcircuitDefinition:
    name: str
    pins: list[str]        # ordered pin names
    definition: str        # full .SUBCKT ... .ENDS block
    source_file: Path

@dataclass
class LoadedLibrary:
    path: Path
    models: dict[str, ModelDefinition] = field(default_factory=dict)
    subcircuits: dict[str, SubcircuitDefinition] = field(default_factory=dict)

    @property
    def include_directive(self) -> str:
        return f".include {self.path}"

def parse_library(path: Path) -> LoadedLibrary:
    """Parse a text file for .MODEL and .SUBCKT definitions.

    For .SUBCKT: enumerate names with regex, then delegate to
    SpiceCircuit.find_subckt_in_lib() for full parsing (pin extraction,
    nested subcircuit handling).

    For .MODEL: custom regex extraction (spicelib has no .MODEL parser).
    """
    ...
```

The `.MODEL` parser handles:
- `.MODEL name type(params...)` — single or multi-line (continuation with `+`)
- Case-insensitive matching (SPICE convention)
- Comment stripping (`*` and `;` prefixed lines)

The `.SUBCKT` enumeration scans for names, then uses spicelib's
`find_subckt_in_lib()` to parse each one — getting pin lists and full
definitions for free without reimplementing that logic.

### Simulation Job Tracking

```python
@dataclass
class SimulationJob:
    job_id: str
    netlist: Path
    simulator: str
    status: Literal["queued", "running", "completed", "failed", "timeout"]
    started_at: datetime
    completed_at: datetime | None
    raw_file: Path | None
    log_file: Path | None
    error: str | None
    task: RunTask | None
    done_event: asyncio.Event  # set when SimRunner callback fires
```

### Error Handling

```python
class LTSpiceMCPError(Exception):
    """Base for all ltspice-mcp errors."""

class PathSecurityError(LTSpiceMCPError):
    """Path resolves outside the working directory."""

class NetlistError(LTSpiceMCPError):
    """Invalid netlist or component reference."""

class SimulationError(LTSpiceMCPError):
    """Simulation execution failed."""

class ConvergenceError(SimulationError):
    """Time step too small / failed to converge."""

class SingularMatrixError(SimulationError):
    """Singular matrix — floating node or short circuit."""

class MissingModelError(SimulationError):
    """Referenced subcircuit or model not found."""

class ResultError(LTSpiceMCPError):
    """Error reading simulation results."""
```

### Log Error Patterns

| Log Pattern | Error Type | Suggested Action |
|---|---|---|
| "Time step too small" | ConvergenceError | Increase max timestep, add `.options`, simplify model |
| "Singular matrix" | SingularMatrixError | Check for floating nodes, verify ground connection |
| "Unknown subcircuit" | MissingModelError | Report missing model name, suggest alternatives |
| "Can't find" | MissingModelError | Report missing file, check include paths |
| "Syntax error" | NetlistError | Report line number and content |
| "Analysis: interrupted" | SimulationError | Timeout or user abort |

## 10. Configuration

```python
@dataclass
class ServerConfig:
    simulator: str | None          # LTSPICE_SIMULATOR env var or auto-detect
    simulator_exe: Path | None     # LTSPICE_EXE env var (override auto-detect)
    working_dir: Path              # LTSPICE_WORKDIR or ./circuits/
    max_parallel_sims: int         # LTSPICE_MAX_PARALLEL or 4
    default_timeout: float         # LTSPICE_TIMEOUT or 300.0
    max_points_returned: int       # Cap on waveform data points (default: 10000)
    plot_dpi: int                  # default: 150
    plot_style: str                # default: "seaborn-v0_8-darkgrid"
```

Env vars:
- `LTSPICE_SIMULATOR`: `"ltspice"`, `"ngspice"`, `"qspice"`, `"xyce"`
- `LTSPICE_EXE`: path to simulator executable (skips auto-detection)
- `LTSPICE_WORKDIR`: working directory for circuit files
- `LTSPICE_MAX_PARALLEL`: max concurrent simulations
- `LTSPICE_TIMEOUT`: default simulation timeout in seconds

Per-simulator env vars (used by spicelib's own detection):
- `LTSPICEFOLDER`, `LTSPICEEXECUTABLE`: LTSpice on Linux/macOS

## 11. Response Format Conventions

### Text + Structured Data

```python
async def handle_get_measurements(args: dict) -> list[types.Content]:
    # ... parse log ...
    text = "Measurements:\n"
    text += f"  vout_max = 3.28V\n"
    text += f"  rise_time = 1.2us\n"
    return [types.TextContent(type="text", text=text)]
```

### Images + Text Summary

```python
async def handle_plot_bode(args: dict) -> list[types.Content]:
    png_bytes, summary = render_bode_plot(...)
    return [
        types.ImageContent(
            type="image",
            data=base64.b64encode(png_bytes).decode(),
            mimeType="image/png",
        ),
        types.TextContent(
            type="text",
            text=f"Bode plot of {trace}:\n"
                 f"  DC gain: {summary['dc_gain_db']:.1f} dB\n"
                 f"  -3dB bandwidth: {format_eng(summary['bandwidth'])}Hz\n"
                 f"  Phase margin: {summary['phase_margin']:.1f} deg",
        ),
    ]
```

### Waveform Data Truncation

If a trace has >10k points:
1. Auto-downsample to `max_points_returned` via linear interpolation
2. Note in the response: "Downsampled from 150,000 to 10,000 points"
3. `downsample` parameter lets the caller override

## 12. Dependencies

Managed with `uv`. The project uses `uv init`, `uv add`, and `uv run`
throughout development.

```toml
[project]
name = "ltspice-mcp"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "mcp>=1.2.0",            # MCP SDK (low-level Server API)
    "spicelib>=1.4.9",        # SPICE simulator automation (brings numpy, matplotlib)
]

[project.scripts]
ltspice-mcp = "ltspice_mcp.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ltspice_mcp"]
```

Notes:
- `spicelib` already depends on `matplotlib` and `numpy` — no need to list
  them separately.
- `mcp` brings `pydantic` — no need to list it separately.
- The `[tool.hatch.build.targets.wheel]` section is required for the `src/`
  layout.
- Run with: `uv run ltspice-mcp` or `uv run python -m ltspice_mcp`.

## 13. Implementation Order

### Phase 1: Skeleton + Configuration

**Goal**: server starts, reports available simulators, accepts MCP connections.

**Deliverable**: `uv run ltspice-mcp` starts cleanly, MCP inspector connects
and receives an empty tool list.

1. **Restructure to src/ layout**
   - Delete root `main.py`
   - Create full `src/ltspice_mcp/` directory tree (Section 9)
   - Create all `__init__.py` files

2. **Update `pyproject.toml`**
   - Add `mcp>=1.2.0` dependency
   - Add `[project.scripts]` entry point: `ltspice-mcp = "ltspice_mcp.main:main"`
   - Add `[tool.hatch.build.targets.wheel]` for src/ layout
   - `uv sync` to install mcp

3. **Core modules (independent, implement in any order)**
   - `errors.py` — full error hierarchy including `PathSecurityError`
   - `config.py` — `ServerConfig.from_env()` reading all env vars
   - `lib/pathutil.py` — `resolve_safe_path()`
   - `lib/cache.py` — generic `FileCache[T]`
   - `lib/format.py` — engineering notation helpers (stub initially)

4. **Simulator detection**
   - `lib/simulator.py` — `detect_simulators()` probing all 4 simulator classes

5. **Session state**
   - `state.py` — `SessionState` dataclass with `create()` and `shutdown()`
   - `SimulationJob` dataclass with `done_event`

6. **Tool infrastructure**
   - `tools/_base.py` — `safe_path()` wrapping pathutil, `run_sync()` wrapping
     `asyncio.to_thread()`
   - Stub tool modules (`circuit.py`, `simulation.py`, `analysis.py`,
     `visualization.py`, `advanced.py`, `library.py`) each exporting empty
     `TOOL_DEFS = []` and `TOOL_HANDLERS = {}`

7. **Server assembly**
   - `server.py` — lifespan, `Server` instance, `list_tools()` / `call_tool()`
     dispatch table collecting from all tool modules
   - `main.py` — `main()` entry point with stdio transport
   - `__main__.py` — enables `python -m ltspice_mcp`

### Phase 2: Circuit Management Tools (10 tools)

**Goal**: LLM can create, read, and modify SPICE netlists.

**Deliverable**: all 10 circuit tools work end-to-end — create a netlist,
read it back, tweak component values, verify changes persist on disk.

**Files**: `tools/circuit.py`, `tools/_base.py`

Tools in implementation order:
1. `create_netlist` — write content string to `.cir` in working_dir
2. `read_netlist` — read file, parse with `SpiceEditor`, return content +
   component list
3. `list_components` — `SpiceEditor.get_components()` with prefix filter
4. `get_component_value` — `SpiceEditor.get_component_value()`
5. `set_component_value` — edit in-memory + `save_netlist()` via `to_thread()`
6. `set_component_values` — batch version of above
7. `get_parameters` — `get_all_parameter_names()` + `get_parameter()` loop
8. `set_parameter` — `SpiceEditor.set_parameter()`
9. `add_instruction` / `remove_instruction` — directive management
10. `convert_schematic` — `.asc` to `.net` via LTSpice (LTSpice-only)

All path arguments validated via `safe_path()`. All file writes via
`asyncio.to_thread()`. `SpiceEditor` instances cached in `state.editors`.

### Phase 3: Simulation Execution (4 tools)

**Goal**: LLM can run simulations synchronously or asynchronously and check
status.

**Deliverable**: run an RC filter simulation, get back raw/log file paths,
verify error parsing works for common failures.

**Files**: `tools/simulation.py`, `state.py`, `lib/simulator.py`

1. `run_simulation` — `asyncio.to_thread(runner.run_now, ...)` with timeout.
   After completion, scan log for error patterns (Section 9, Log Error
   Patterns) and raise typed errors with actionable suggestions.
2. `run_simulation_async` — `SimRunner.run()` with callback bridge:
   callback fires in worker thread, sets job fields, calls
   `loop.call_soon_threadsafe(job.done_event.set)`. Returns job_id.
3. `get_simulation_status` — look up job in `state.jobs`, optionally wait
   briefly on `done_event` before returning current status.
4. `list_simulations` — iterate `state.jobs`, return summary array.

### Phase 4: Result Analysis (7 tools)

**Goal**: LLM can extract numeric data from simulation results.

**Deliverable**: after a simulation, get trace names, extract waveform data
(with downsampling), read .MEAS results, get operating point.

**Files**: `tools/analysis.py`

1. `get_trace_names` — `RawRead` header-only load (`traces_to_read=None`),
   return trace metadata
2. `get_simulation_summary` — combine raw header + log parsing into overview
3. `get_waveform_data` — load specific traces via `RawRead`, auto-downsample
   to `config.max_points_returned` (10k default), handle AC analysis
   (return magnitude/phase instead of complex)
4. `get_measurements` — `LTSpiceLogReader` for `.MEAS` results, indexed by
   step for stepped sims
5. `get_operating_point` — parse `.op` raw file, separate node voltages from
   branch currents
6. `get_node_voltage` — single-point lookup with numpy interpolation at
   specific time/frequency
7. `get_fourier_data` — `LTSpiceLogReader.fourier` extraction

`RawRead` instances cached via `state.results` `FileCache`. All file I/O
via `asyncio.to_thread()`.

### Phase 5: Visualization (4 tools + plotting library)

**Goal**: LLM gets inline PNG plots of waveforms and Bode diagrams.

**Deliverable**: generate waveform and Bode plots from simulation data,
receive valid PNG + text summary with min/max/bandwidth numbers.

**Files**: `tools/visualization.py`, `lib/plotting.py`

1. **`lib/plotting.py`** — pure sync rendering functions:
   - `render_waveform(traces, axis, title) -> tuple[bytes, dict]`
   - `render_bode(trace, axis) -> tuple[bytes, dict]` — dual subplot
   - `render_histogram(values, name, bins) -> tuple[bytes, dict]`
   - `render_xy(x_data, y_data, labels) -> tuple[bytes, dict]`
   - Each returns PNG bytes + summary dict (min/max/mean/bandwidth/margin)
   - Shared style: `config.plot_style`, `config.plot_dpi`

2. Tool handlers call analysis helpers to load data, then render via
   `asyncio.to_thread()`, return `[ImageContent, TextContent]`.

### Phase 6: Component Library + Library Management (7 tools)

**Goal**: LLM can search built-in libraries and load custom library files.

**Deliverable**: load a `.lib` file, list its models/subcircuits, look up a
specific component, get the `.include` directive to use in a netlist.

**Files**: `tools/library.py`, `lib/library_parser.py`

1. **`lib/library_parser.py`**:
   - `parse_library(path) -> LoadedLibrary`
   - `.SUBCKT` enumeration: regex scan for names, then delegate to
     `SpiceCircuit.find_subckt_in_lib()` for full parsing
   - `.MODEL` extraction: custom regex (spicelib has no `.MODEL` parser)
   - Handles multi-line continuations (`+`), case-insensitive, comment stripping

2. Built-in library tools:
   - `search_components` — scan simulator library paths + loaded libraries
   - `get_model_info` — point lookup in library files
   - `list_subcircuits` — list from simulator library paths

3. Library management tools:
   - `load_library` — parse file, store in `state.libraries`, return summary
   - `get_library_component` — lookup across loaded libraries, return
     definition + `include_directive`
   - `list_loaded_libraries` — iterate `state.libraries`
   - `unload_library` — remove from `state.libraries`

### Phase 7: Advanced Analysis (4 tools)

**Goal**: LLM can run parameter sweeps and Monte Carlo analysis.

**Deliverable**: configure a sweep, run batch, query aggregated results
with statistics.

**Files**: `tools/advanced.py`

1. `setup_parameter_sweep` — configure `SimStepper` on a netlist copy
2. `setup_monte_carlo` — configure `Montecarlo` with tolerances/distribution
3. `run_batch_analysis` — execute sweep/MC via `SimRunner`, track all jobs,
   return completion summary
4. `get_sweep_results` — aggregate measurements from logs, compute
   min/max/mean/std, support parameter-value filtering

### Phase 8: Resources + Prompts

**Goal**: MCP resources for browsing netlists/results, prompt templates for
guided workflows.

**Deliverable**: resources return correct data when queried, prompts produce
structured guidance text.

**Files**: `resources.py`, `prompts.py`, `server.py` (register handlers)

1. Resources (Section 7): `ltspice://netlists`, `ltspice://netlists/{name}`,
   `ltspice://results/{name}/summary`, `ltspice://results/{name}/log`,
   `ltspice://library/models`, `ltspice://config`
2. Prompts (Section 8): `design_filter`, `analyze_amplifier`,
   `tolerance_analysis`, `debug_simulation`
3. Register resource/prompt handlers in `server.py`

### Verification

After each phase, verify the deliverable described above. End-to-end test
after all phases complete:

1. Create RC lowpass netlist (R=1k, C=100n → fc ~1.59kHz)
2. Add `.ac dec 100 1 1Meg` directive
3. Run AC simulation
4. Get Bode plot → verify -3dB point near 1.59kHz
5. Change R to 10k (fc → ~159Hz)
6. Re-simulate, re-plot → verify bandwidth shifted
7. Load a custom library, use a component from it in a new circuit

## 14. Testing Strategy

- **Unit tests**: mock `SpiceEditor`, `RawRead`, `LTSpiceLogReader` to test
  tool handlers in isolation. Use sample `.raw` and `.log` files.
- **Integration tests**: require a simulator installed. Run actual simulations
  on simple circuits (RC filter, voltage divider). Mark with
  `@pytest.mark.integration`. Parameterize across available simulators.
- **MCP protocol tests**: use the SDK's `ClientSession` to connect to the
  server in-process and verify tool schemas, request/response format.
- **Snapshot tests**: for visualization, compare generated PNGs against
  reference images (pixel-diff with tolerance).

## 15. Open Questions (Resolved)

1. **Schematic creation**: Netlist-only for v1. spicelib can modify existing
   `.asc` files but not create them from scratch; schematics are mainly for
   GUI editing and are not needed for simulation.

2. **Waveform data format**: JSON arrays with auto-downsampling. An optional
   `export_csv` tool for full-resolution data can be added later if needed.

3. **Multi-simulator result comparison**: Defer to v2.

4. **QSPICE `.qraw` handling**: `RawRead` handles dialect detection, and
   `SimRunner` picks up `raw_extension` from the `Qspice` class. Verify
   end-to-end during Phase 3 integration testing.

5. **NGspice compatibility mode**: Default to `"kiltpsa"` (spicelib default).
   Expose as `NGSPICE_COMPAT_MODE` env var in `ServerConfig`.
